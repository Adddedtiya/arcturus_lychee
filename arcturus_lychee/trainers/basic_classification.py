"""The training loop for image classification. This file is the worked example.

For a new project, copy this file. Then change train_step(), eval_step(),
and report(). The other methods usually stay the same.

Rules for this file:

  - Settings go in the configuration. Read them with self.configuration.
  - State goes on self. Make all state in __init__.
  - Do not put variables at file level. Each process imports this file again,
    so a change to a file-level variable does not get to the other processes.

Under DDP (two or more GPUs), DistributedDataParallel synchronizes the
gradients in each backward pass. Rank 0 does the evaluation, the logs, and
the checkpoints. The training metrics include all ranks. With one GPU, the
same code operates without a process group.
"""

import torch
import torch.nn as nn

from torch.optim              import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data         import DataLoader
from tqdm                     import tqdm
from typing                   import Optional, Union

from arcturus_lychee.configuration import TrainingConfiguration, toml_compatible
from arcturus_lychee.helpers import (
    DirectoryTrainingLogger,
    NullLogger,
    SpeedTimer,
    MetricAccumulator,
    TimedLoader,
    Precision,
    wrap_model,
    unwrap_model,
    move_to_device,
    set_sampler_epoch,
    get_device,
    get_world_size,
    is_main_process,
    barrier,
    all_reduce_sum_,
    all_reduce_max,
    generate_report,
    generate_confusion_matrix,
)


def top_k_accuracy(
        prediction : torch.Tensor,
        target     : torch.Tensor,
        ks         : tuple[int, ...],
    ) -> dict[str, torch.Tensor]:
    """Return the top-k accuracy for each k, as tensors on the device.

    A value of k that is more than the number of classes is not in the result.
    """
    ks = tuple(k for k in ks if k <= prediction.size(1))
    if not ks:
        return {}

    _, predicted = prediction.topk(max(ks), dim = 1)      # [batch, max_k]
    correct      = predicted.eq(target.view(-1, 1))       # [batch, max_k]
    return {f"top-{k}": correct[:, :k].any(dim = 1).float().mean() for k in ks}


def format_metrics(metrics : dict[str, float]) -> str:
    """Return the metrics as one line of text, for example "loss 0.4123, top-1 0.8125"."""
    return ", ".join(f"{name} {value:.4g}" for name, value in metrics.items())


class ClassificationTrainer:
    """Train, evaluate, and save an image classification model.

    Edit these methods for a new project:

      - train_step: one training batch
      - eval_step: one evaluation batch
      - report: the output at the end of the training

    These methods usually stay the same:

      - train_epoch, evaluate, fit, save_state, load_state
    """

    def __init__(
            self,
            model         : nn.Module,
            configuration : TrainingConfiguration,
            logger        : Union[DirectoryTrainingLogger, NullLogger, None] = None,
        ) -> None:

        self.configuration = configuration
        self.log           = logger if logger is not None else NullLogger()

        # processes and device
        self.world_size = get_world_size()
        self.is_main    = is_main_process()
        self.device     = get_device()

        self.total_epochs = configuration.total_epochs

        # model: self.model can have a DDP wrapper. self.raw_model never has one.
        self.model = wrap_model(
            model,
            self.device,
            sync_batchnorm         = configuration.use_sync_batchnorm,
            find_unused_parameters = configuration.find_unused_parameters,
        )
        self.raw_model = unwrap_model(self.model)

        # optimizer and scheduler
        learning_rate = configuration.learning_rate
        if configuration.scale_lr_by_world_size and self.world_size > 1:
            scaled = learning_rate * self.world_size
            self.log.log(
                f"The trainer multiplies the learning rate by the number of GPUs "
                f"({self.world_size}): {learning_rate} -> {scaled}"
            )
            learning_rate = scaled

        self.optimizer = AdamW(self.model.parameters(), lr = learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max = self.total_epochs)
        self.criterion = nn.CrossEntropyLoss()
        self.precision = Precision(configuration.precision, self.device)

        self.log.log(f"Device: {self.device}. Precision: {self.precision.name}. Processes: {self.world_size}.")

        # the record of the run
        self.log.record.save_model(self.raw_model)
        skipped = self.log.record.save_optimizer(self.optimizer, self.scheduler)
        if skipped:
            self.log.log(f"NOTE: optimizer.toml does not contain these values: {', '.join(skipped)}.")

    # ----------------------------------------------------------------------- #
    # Edit these methods for a new project
    # ----------------------------------------------------------------------- #

    def train_step(self, batch : tuple[torch.Tensor, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
        """Train on one batch. Return the metrics and the number of samples in the batch."""
        images, labels = move_to_device(batch, self.device)

        with self.precision.autocast():
            prediction = self.model(images)
            loss       = self.criterion(prediction, labels)

        self.precision.backward(loss)
        self.precision.step(self.optimizer)
        self.optimizer.zero_grad(set_to_none = True)

        metrics = {"loss": loss.detach()}
        metrics.update(top_k_accuracy(prediction.detach(), labels, ks = (1, 5)))
        return metrics, images.size(0)

    @torch.no_grad()
    def eval_step(self, batch : tuple[torch.Tensor, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
        """Evaluate one batch. Return the metrics and the number of samples in the batch."""
        images, labels = move_to_device(batch, self.device)

        with self.precision.autocast():
            prediction = self.raw_model(images)
            loss       = self.criterion(prediction, labels)

        metrics = {"loss": loss}
        metrics.update(top_k_accuracy(prediction, labels, ks = (1, 5)))
        return metrics, images.size(0)

    @torch.no_grad()
    def report(
            self,
            loader      : DataLoader,
            title       : str,
            class_names : Optional[list[str]] = None,
        ) -> dict[str, float]:
        """Write the accuracy, the classification report, and the confusion matrix.

        The text goes to the log and to record/report_<title>.txt.
        Only rank 0 makes the report. The other ranks return an empty dict.
        """
        if not self.is_main:
            return {}

        self.raw_model.eval()
        meter     = MetricAccumulator()
        truth     : list[int] = []
        predicted : list[int] = []

        for batch in tqdm(loader, desc = f"Report {title}"):
            images, labels = move_to_device(batch, self.device)
            with self.precision.autocast():
                prediction = self.raw_model(images)

            for name, value in top_k_accuracy(prediction, labels, ks = (1, 3, 5)).items():
                meter.add(name, value, weight = images.size(0))
            truth     += labels.tolist()
            predicted += prediction.argmax(dim = 1).tolist()

        accuracy = meter.reduce(across_ranks = False)

        lines  = [f"Report: {title}", ""]
        lines += [f"Accuracy {name} : {value:.4f}" for name, value in accuracy.items()]
        lines += ["", "Classification report:"]
        lines += generate_report(truth, predicted, class_names)
        lines += ["", "Confusion matrix:"]
        lines += generate_confusion_matrix(truth, predicted, class_names)

        self.log.log(lines)
        self.log.record.save_text(f"report_{title.lower().replace(' ', '_')}.txt", lines)
        return accuracy

    # ----------------------------------------------------------------------- #
    # These methods usually stay the same
    # ----------------------------------------------------------------------- #

    def train_epoch(self, loader : DataLoader, epoch : int) -> dict[str, float]:
        """Train for one epoch. Return the mean metrics of all ranks and the times.

        All ranks must call this method. It uses collective operations.
        """
        self.model.train()
        set_sampler_epoch(loader, epoch)

        timer   = SpeedTimer()
        timed   = TimedLoader(loader, sync_cuda = self.configuration.time_with_cuda_sync)
        meter   = MetricAccumulator()
        samples = 0

        for batch in tqdm(timed, desc = "Train", disable = not self.is_main):
            metrics, batch_samples = self.train_step(batch)
            for name, value in metrics.items():
                meter.add(name, value, weight = batch_samples)
            samples += batch_samples

        stats = meter.reduce(across_ranks = True)

        # The slowest rank sets the time. The samples of all ranks add together.
        times         = all_reduce_max({**timed.stats(), "time_train_s": timer.stop()})
        total_samples = all_reduce_sum_(torch.tensor(float(samples), device = self.device)).item()

        stats.update(times)
        stats["samples_per_s"] = total_samples / times["time_train_s"] if times["time_train_s"] > 0 else 0.0
        return stats

    def evaluate(self, loader : DataLoader) -> dict[str, float]:
        """Evaluate on rank 0 only. Return the mean metrics.

        The method does not use collective operations. Thus the other ranks can
        wait at the barrier in fit().
        """
        self.raw_model.eval()
        meter = MetricAccumulator()

        for batch in tqdm(loader, desc = "Evaluation", disable = not self.is_main):
            metrics, batch_samples = self.eval_step(batch)
            for name, value in metrics.items():
                meter.add(name, value, weight = batch_samples)

        return meter.reduce(across_ranks = False)

    def fit(
            self,
            train_loader : DataLoader,
            eval_loader  : Optional[DataLoader],
            start_epoch  : int = 0,
        ) -> None:
        """Train for all epochs. Evaluate, log, and save checkpoints on rank 0.

        Evaluation: after each test_every_n epochs, and after the last epoch.
        latest.pt: after each save_every_n epochs, and after the last epoch.
        best.pt: after each evaluation with a new best value.
        final.pt and record/summary.toml: at the end.

        To continue a run, see load_state().
        """
        configuration = self.configuration
        total_timer   = SpeedTimer()
        train_stats   : dict[str, float]           = {}
        last_eval     : Optional[dict[str, float]] = None

        self.log.log("The training starts.")

        for epoch in range(start_epoch, self.total_epochs):
            epoch_timer = SpeedTimer()
            is_last     = epoch == self.total_epochs - 1
            self.log.log(f"Epoch {epoch + 1} of {self.total_epochs}")

            # all ranks: train, then change the learning rate
            train_stats       = self.train_epoch(train_loader, epoch)
            train_stats["lr"] = float(sum(self.scheduler.get_last_lr()) / len(self.scheduler.get_last_lr()))
            self.scheduler.step()

            # rank 0: evaluate
            eval_stats  = None
            should_eval = (epoch + 1) % configuration.test_every_n == 0 or is_last
            if self.is_main and eval_loader is not None and should_eval:
                eval_timer = SpeedTimer()
                eval_stats = self.evaluate(eval_loader)
                last_eval  = eval_stats
                train_stats["time_eval_s"] = eval_timer.stop()
            else:
                train_stats["time_eval_s"] = 0.0

            # rank 0: logs and checkpoints
            if self.is_main:
                train_stats["time_epoch_s"] = epoch_timer.elapsed()
                self.log.append(train_stats, eval_stats, epoch = epoch + 1)

                self.log.log(f"Train: {format_metrics(train_stats)}")
                if eval_stats is not None:
                    self.log.log(f"Evaluation: {format_metrics(eval_stats)}")

                if self.log.is_best():
                    self.log.log(f"New best {configuration.metric_to_track}: {self.log.best_value:.4f}. The trainer saves best.pt.")
                    self.save_state(self.log.get_weights_path("best.pt"), epoch = epoch)

                if (epoch + 1) % configuration.save_every_n == 0 or is_last:
                    self.save_state(self.log.get_weights_path("latest.pt"), epoch = epoch)

                self.log.log(SpeedTimer.estimate_time(epoch_timer, self.total_epochs - epoch - 1))

            # All ranks wait here. Rank 0 can need more time for the evaluation and the checkpoints.
            barrier()

        if self.is_main:
            self.save_state(self.log.get_weights_path("final.pt"), epoch = self.total_epochs - 1)
            self.log.record.save_toml("summary.toml", {
                "run_name"     : self.log.run_name,
                "total_epochs" : self.total_epochs,
                "total_time_s" : total_timer.stop(),
                "best"         : {
                    "metric" : configuration.metric_to_track,
                    "value"  : self.log.best_value,
                    "epoch"  : self.log.best_epoch,
                },
                "last_train"   : train_stats,
                "last_eval"    : last_eval or {},
            })

        self.log.log("The training is complete.")

    # ----------------------------------------------------------------------- #
    # Checkpoints
    # ----------------------------------------------------------------------- #

    def save_state(self, fpath : str, epoch : int = 0) -> None:
        """Save a checkpoint. The model has no DDP wrapper, so the file loads with and without DDP.

        The checkpoint also contains the configuration, as a dict of plain values.
        """
        configuration, _ = toml_compatible(vars(self.configuration))
        state = {
            'epoch'           : int(epoch),
            'model_state'     : self.raw_model.state_dict(),
            'optimizer_state' : self.optimizer.state_dict(),
            'scheduler_state' : self.scheduler.state_dict(),
            'scaler_state'    : self.precision.state_dict(),
            'configuration'   : configuration,
        }
        torch.save(state, fpath)

    def load_state(self, fpath : str) -> int:
        """Load a checkpoint, and return its epoch index (the first epoch is 0).

        Each rank can call this method. Each rank loads the data to its own device.

        To continue a run (README section 8). All ranks load the checkpoint, so
        the path comes from the configuration, not from the logger:

            configuration = load_run_configuration("<run directory>")
            configuration.prefix_date     = False
            configuration.experiment_name = configuration.run_name

            latest     = os.path.join(configuration.working_directory, configuration.run_name, "weights", "latest.pt")
            logger     = DirectoryTrainingLogger(configuration) if is_main_process() else NullLogger()
            trainer    = ClassificationTrainer(model, configuration, logger)
            last_epoch = trainer.load_state(latest)
            logger.load_from_csv()
            trainer.fit(train_loader, eval_loader, start_epoch = last_epoch + 1)
        """
        state = torch.load(fpath, map_location = self.device, weights_only = True)

        self.raw_model.load_state_dict(state['model_state'], strict = True)
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.precision.load_state_dict(state['scaler_state'])

        if state.get('scheduler_state') is not None:
            self.scheduler.load_state_dict(state['scheduler_state'])

        return int(state.get('epoch', 0))
