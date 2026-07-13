import torch
import torch.nn             as nn

from torch.optim                  import AdamW
from torch.optim.lr_scheduler     import CosineAnnealingLR
from torch.amp.grad_scaler        import GradScaler
from torch.utils.data             import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel            import DistributedDataParallel as DDP

from arcturus_lychee.helpers import (
    DirectoryTrainingLogger,
    NullLogger,
    SpeedTimer,
    generate_report,
    generate_confusion_matrix,
)
from arcturus_lychee.helpers.distributed import (
    get_world_size,
    is_main_process,
    barrier,
    all_reduce_metric_sums,
)
from arcturus_lychee.configuration.basic_template import TrainingConfiguration

from collections import defaultdict
from tqdm        import tqdm
from typing      import Union


class WrapperForClassification:
    """Classification training loop with transparent single-/multi-GPU support.

    Under DDP (``world_size > 1``) the model is wrapped in
    ``DistributedDataParallel`` and gradients are synchronised every backward
    pass. Everything that is not a training collective - evaluation, the sklearn
    report, checkpointing and all disk logging - runs on rank 0 only. The
    training-metric averages are all-reduced so the logged curves reflect the
    whole dataset rather than rank 0's shard.

    When ``world_size == 1`` no process group exists, every distributed helper
    no-ops, and the behaviour is identical to the original single-GPU trainer.

    The configuration is passed in directly (rather than read off the logger),
    because every rank needs the config but only rank 0 owns a real logger.
    """

    def __init__(
            self,
            model         : nn.Module,
            configuration : TrainingConfiguration,
            logger        : Union[DirectoryTrainingLogger, NullLogger, None] = None,
        ):

        # config + logger (logger is None / NullLogger on non-main ranks)
        self.configuration = configuration
        self.log           = logger if logger is not None else NullLogger()

        # distributed context
        self.world_size = get_world_size()
        self.is_main    = is_main_process()

        # Per-rank device. Under DDP each rank owns exactly one GPU; use the
        # device the process group pinned. `.type` (not str(device)) is used for
        # autocast / GradScaler so an indexed device like cuda:1 works correctly.
        if torch.cuda.is_available() and self.world_size > 1:
            self.device = torch.device("cuda", torch.cuda.current_device())
        else:
            self.device = configuration.device

        # dtype is resolved lazily here (inside the worker), never in the parent
        self.data_type = configuration.resolved_dtype()

        # epochs
        self.total_epochs = configuration.total_epochs

        # ---- model placement, optional SyncBN, optional DDP wrap ----
        model = model.to(self.device)

        if self.world_size > 1 and configuration.use_sync_batchnorm:
            # fuse BatchNorm stats across GPUs - useful when the per-GPU batch is
            # small (must happen before the DDP wrap)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if self.world_size > 1:
            device_ids = [self.device.index] if self.device.type == "cuda" else None
            self.model = DDP(
                model,
                device_ids            = device_ids,
                output_device         = self.device.index if self.device.type == "cuda" else None,
                find_unused_parameters = configuration.find_unused_parameters,
            )
        else:
            self.model = model

        # unwrapped reference: used for eval/test forward and for checkpoints so
        # saved state_dicts have no "module." prefix and stay single-GPU loadable
        self.raw_model = self.model.module if isinstance(self.model, DDP) else self.model

        # ---- optimizer (with optional linear LR scaling) ----
        learning_rate = configuration.learning_rate
        if configuration.scale_lr_by_world_size and self.world_size > 1:
            scaled = learning_rate * self.world_size
            self.log.print(f"[DDP] Scaling LR by world_size={self.world_size}: {learning_rate} -> {scaled}")
            learning_rate = scaled

        self.optimizer = AdamW(self.model.parameters(), lr = learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max = self.total_epochs)

        # loss
        self.criterion = nn.CrossEntropyLoss()

        # AMP: autocast on CUDA; GradScaler only for true fp16 (bf16/fp32 skip it)
        self.use_amp         = self.device.type == "cuda"
        self.use_grad_scaler = self.data_type == torch.float16
        self.gradient_scaler = GradScaler(self.device.type, enabled = self.use_grad_scaler)

    # ----------------------------------------------------------------------- #
    # Checkpointing
    # ----------------------------------------------------------------------- #

    def save_state(self, fpath : str, epoch : int = 0) -> None:
        # save the UNWRAPPED model so checkpoints load with or without DDP
        state = {
            'epoch'           : int(epoch),
            'model_state'     : self.raw_model.state_dict(),
            'optimizer_state' : self.optimizer.state_dict(),
            'scheduler_state' : self.scheduler.state_dict(),
            'scaler_state'    : self.gradient_scaler.state_dict(),
        }
        torch.save(state, fpath)

    def load_state(self, fpath : str) -> int:
        """Restore model / optimizer / scheduler / scaler and return the saved epoch.

        Safe to call on every rank when resuming: each maps the checkpoint onto
        its own device.

        To resume training:
            last_epoch = wrapper.load_state(logger.get_weights_path("latest.pt"))
            logger.load_from_csv()                      # restore metric history + best value + epoch
            wrapper.run_everything(..., start_epoch = last_epoch + 1)
        """
        state = torch.load(fpath, map_location = self.device, weights_only = True)

        self.raw_model.load_state_dict(state['model_state'], strict = True)
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.gradient_scaler.load_state_dict(state['scaler_state'])

        if state.get('scheduler_state') is not None:
            self.scheduler.load_state_dict(state['scheduler_state'])

        return int(state.get('epoch', 0))

    # ----------------------------------------------------------------------- #
    # Metrics
    # ----------------------------------------------------------------------- #

    def __compute_top_n(self, x_real : torch.Tensor, y_pred : torch.Tensor, top_k : tuple[int, ...] = (1, 5)) -> dict[str, float]:

        y_pred = y_pred.detach()
        results = {}

        with torch.no_grad():
            # clamp k to the number of classes so e.g. top-5 on a few-class model
            # doesn't raise "selected index k out of range"
            num_classes = y_pred.size(1)
            top_k = tuple(k for k in top_k if k <= num_classes)
            if not top_k:
                return results

            max_k      = max(top_k)
            batch_size = x_real.size(0)

            _, pred = y_pred.topk(max_k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(x_real.view(1, -1).expand_as(pred))

            for k in top_k:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
                results[f"top-{k}"] = float(correct_k.item() / batch_size)

        return results

    def __compute_metrics(self, x_real : torch.Tensor, y_pred : torch.Tensor) -> dict[str, float]:
        return self.__compute_top_n(x_real, y_pred, top_k = (1, 5))

    def __calculate_metric_averages(
            self,
            metrics             : list[dict[str, float]],
            weights             : Union[list[float], None] = None,
            reduce_across_ranks : bool = False,
        ) -> dict[str, float]:

        if not metrics and not reduce_across_ranks:
            return {}

        if weights is None:
            weights = [1.0] * len(metrics)

        weighted_sum  = defaultdict(float)
        weight_totals = defaultdict(float)
        for item, weight in zip(metrics, weights):
            for key, value in item.items():
                weighted_sum[key]  += value * weight
                weight_totals[key] += weight

        # sum the per-rank totals so the mean is over the whole dataset, not one shard
        if reduce_across_ranks:
            weighted_sum, weight_totals = all_reduce_metric_sums(dict(weighted_sum), dict(weight_totals))

        return {
            key: float(weighted_sum[key] / weight_totals[key])
            for key in weighted_sum
            if weight_totals.get(key, 0.0) != 0.0
        }

    # ----------------------------------------------------------------------- #
    # Single-batch steps
    # ----------------------------------------------------------------------- #

    def __train_single_batch(self, input_tensor : torch.Tensor, target_tensor : torch.Tensor) -> dict[str, float]:

        self.model.train()

        with torch.autocast(device_type = self.device.type, dtype = self.data_type, enabled = self.use_amp):
            input_tensor  = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)

            prediction : torch.Tensor = self.model(input_tensor)
            loss : torch.Tensor = self.criterion(prediction, target_tensor)

        scaled_loss = self.gradient_scaler.scale(loss)
        scaled_loss.backward()

        self.gradient_scaler.step(self.optimizer)
        self.gradient_scaler.update()
        self.optimizer.zero_grad(set_to_none = True)

        metrics = self.__compute_metrics(target_tensor, prediction)
        metrics['loss'] = float(loss.item())
        return metrics

    def __test_single_batch(self, input_tensor : torch.Tensor, target_tensor : torch.Tensor) -> dict[str, float]:

        # eval uses the unwrapped model - no gradient sync, safe to run on rank 0 alone
        self.raw_model.eval()

        with torch.autocast(device_type = self.device.type, dtype = self.data_type, enabled = self.use_amp), torch.no_grad():
            input_tensor  = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)

            prediction : torch.Tensor = self.raw_model(input_tensor)
            metrics = self.__compute_metrics(target_tensor, prediction)

        return metrics

    # ----------------------------------------------------------------------- #
    # Epoch loops
    # ----------------------------------------------------------------------- #

    def _set_train_sampler_epoch(self, dataloader : DataLoader, epoch : int) -> None:
        # DistributedSampler needs set_epoch() each epoch or the shuffle is frozen
        sampler = getattr(dataloader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

    def train_single_epoch(self, dataloader : DataLoader, enable_tqdm : bool = True) -> dict[str, float]:

        active_metrics = []
        batch_sizes    = []
        self.model.train()

        for _, (input_tensor, target_tensor) in enumerate(tqdm(dataloader, disable = not (enable_tqdm and self.is_main))):
            batch_metrics = self.__train_single_batch(input_tensor, target_tensor)
            active_metrics.append(batch_metrics)
            batch_sizes.append(input_tensor.size(0))

        # every rank participates in the reduction (a collective) so the logged
        # training numbers are the true epoch-wide mean
        return self.__calculate_metric_averages(
            active_metrics, batch_sizes, reduce_across_ranks = self.world_size > 1
        )

    def test_single_epoch(self, dataloader : DataLoader, enable_tqdm : bool = True) -> dict[str, float]:

        active_metrics = []
        batch_sizes    = []
        self.raw_model.eval()

        for _, (input_tensor, target_tensor) in enumerate(tqdm(dataloader, disable = not enable_tqdm)):
            batch_metrics = self.__test_single_batch(input_tensor, target_tensor)
            active_metrics.append(batch_metrics)
            batch_sizes.append(input_tensor.size(0))

        # rank-0-only: no cross-rank reduction (avoids DistributedSampler padding
        # double-counting) - see run_single_epoch
        return self.__calculate_metric_averages(active_metrics, batch_sizes)

    def __get_current_learning_rate(self) -> float:
        current_rates = [float(x) for x in self.scheduler.get_last_lr()]
        return float(sum(current_rates) / len(current_rates))

    def run_single_epoch(
            self,
            train_dataloader : DataLoader,
            test_dataloader  : Union[DataLoader, None],
            current_epoch    : int,
            enable_tqdm      : bool = True,
            test_every       : int  = 1,
        ) -> tuple[dict[str, float], Union[dict[str, float], None], SpeedTimer]:

        total_runtime = SpeedTimer()

        # reshuffle this rank's shard for the epoch, then train (all ranks)
        self._set_train_sampler_epoch(train_dataloader, current_epoch)
        train_stats = self.train_single_epoch(train_dataloader, enable_tqdm)

        # evaluation is rank-0 only (keeps the sklearn path exact, no padding dupes)
        test_stats = None
        should_test = (current_epoch % test_every) == 0 or (current_epoch == 0)
        if self.is_main and test_dataloader is not None and should_test:
            test_stats = self.test_single_epoch(test_dataloader, enable_tqdm)

        # log the LR that was active this epoch, then step (all ranks step in lockstep)
        train_stats['lr'] = self.__get_current_learning_rate()
        self.scheduler.step()

        total_runtime.stop()
        return train_stats, test_stats, total_runtime

    def run_everything(
            self,
            train_dataloader : DataLoader,
            test_dataloader  : Union[DataLoader, None],
            enable_tqdm      : bool = True,
            test_every       : int  = 1,
            start_epoch      : int  = 0,
            save_every       : int  = 1,
        ) -> None:

        self.log.print("Starting Training !")

        for current_epoch in range(start_epoch, self.total_epochs):

            self.log.print(f"Current Epoch : {current_epoch + 1}")

            train_stats, eval_stats, total_runtime = self.run_single_epoch(
                train_dataloader = train_dataloader,
                test_dataloader  = test_dataloader,
                current_epoch    = current_epoch,
                enable_tqdm      = enable_tqdm,
                test_every       = test_every,
            )

            # logging / best-model / periodic saves: rank 0 only
            if self.is_main:
                self.log.append(train_stats, eval_stats)

                if self.log.is_best():
                    self.log.print("--- Best model detected! ---")
                    self.save_state(self.log.get_weights_path('best.pt'), epoch = current_epoch)

                if (current_epoch % save_every) == 0 or (current_epoch == 0):
                    self.save_state(self.log.get_weights_path("latest.pt"), epoch = current_epoch)

                leftover_epochs = self.total_epochs - current_epoch
                self.log.print(SpeedTimer.estimate_time(total_runtime, leftover_epochs))

            # keep ranks in lockstep each epoch: rank 0 may have spent extra time
            # on eval / checkpointing while the others waited here
            barrier()

        # final checkpoint: rank 0 only
        if self.is_main:
            self.save_state(self.log.get_weights_path("final.pt"), epoch = self.total_epochs - 1)

        self.log.print("Training is Complete !")

    # ----------------------------------------------------------------------- #
    # Standalone test / report (rank 0 only)
    # ----------------------------------------------------------------------- #

    def test_model(
            self,
            test_dataloader  : DataLoader,
            report_prefix    : str                    = "",
            enable_tqdm      : bool                   = True,
            class_names      : Union[list[str], None] = None,
        ) -> None:

        # no collectives here; run only on rank 0
        if not self.is_main:
            return

        predicted_results   : list[int] = []
        ground_truth_values : list[int] = []
        accuracy_reports    : list[dict[str, float]] = []
        batch_sizes         : list[float]            = []

        self.log.print("")
        self.log.print(f"Testing For : {report_prefix}")

        self.raw_model.eval()
        for _, (input_tensor, target_tensor) in enumerate(tqdm(test_dataloader, disable = not enable_tqdm)):

            with torch.autocast(device_type = self.device.type, dtype = self.data_type, enabled = self.use_amp), torch.no_grad():
                input_tensor  = input_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)

                prediction : torch.Tensor = self.raw_model(input_tensor)

                top_k_values = self.__compute_top_n(target_tensor, prediction, top_k = (1, 3, 5))
                accuracy_reports.append(top_k_values)
                batch_sizes.append(input_tensor.size(0))

            ground_truth_values += target_tensor.detach().cpu().tolist()
            preds = torch.argmax(prediction, dim = 1)
            predicted_results += preds.detach().cpu().tolist()

        average_top_k = self.__calculate_metric_averages(accuracy_reports, batch_sizes)
        for top_k in average_top_k.keys():
            self.log.print(f"Accuracy in {top_k} : {average_top_k[top_k]:.2f}")
        self.log.print("")

        self.log.print("Report :")
        report_lines = generate_report(ground_truth_values, predicted_results, class_names)
        self.log.print_lines(report_lines)
        self.log.print("")

        self.log.print("Confusion Matrix :")
        confusion_lines = generate_confusion_matrix(ground_truth_values, predicted_results, class_names)
        self.log.print_lines(confusion_lines)
        self.log.print("")
