"""Portable multi-/single-GPU training entrypoint.

Run it the same way everywhere - no launcher required:

    python3 main_train_ddp.py

Behaviour is decided at runtime from the visible GPUs:
  * 2+ GPUs  -> one process per GPU (DistributedDataParallel over NCCL), spawned
                internally by `launch()`.
  * 1 GPU / CPU -> a single inline process, no distribution.

Because it is a plain script (not `python -m ...`, not `torchrun`), it runs
identically under SLURM, Docker, or a bare shell. `CUDA_VISIBLE_DEVICES` (e.g.
what SLURM sets from --gres) is honoured automatically, so only allocated GPUs
are used.

`batch_size` in the config is the PER-GPU batch size; the effective global batch
is `batch_size * num_gpus`.
"""

import torch

from arcturus_lychee.configuration                         import TrainingConfiguration
from arcturus_lychee.helpers                               import (
    launch, set_seed, is_main_process,
    DirectoryTrainingLogger, NullLogger,
)
from arcturus_lychee.datasets                              import DirectoryClassification, heavy_aug
from arcturus_lychee.trainers.basic_classification         import WrapperForClassification
from arcturus_lychee.models.architecture.mobile_net        import BasicMobileNetV3


def build_config() -> TrainingConfiguration:
    """Built in the PARENT process. Keep it CUDA-free: do not construct models
    or touch tensors here (dtype stays lazily unresolved on purpose)."""
    cfg = TrainingConfiguration()

    cfg.working_directory = "results"
    cfg.experiment_name   = "ddp_training"

    # dataset roots (edit these) - ImageFolder-style class subdirectories
    cfg.dataset_root_train = "/path/to/dataset/train"
    cfg.dataset_root_val   = "/path/to/dataset/val"
    cfg.dataset_root_test  = "/path/to/dataset/test"

    cfg.total_epochs       = 8
    cfg.batch_size         = 32     # PER-GPU
    cfg.total_workers      = 4
    cfg.model_output_class = 40

    # multi-GPU knobs (conservative defaults; opt in per experiment)
    cfg.scale_lr_by_world_size = False   # set True for linear LR scaling
    cfg.use_sync_batchnorm     = False   # set True if the per-GPU batch is small
    return cfg


def worker(rank: int, world_size: int, cfg: TrainingConfiguration) -> None:
    """Runs once per GPU (or once total on a single GPU / CPU)."""

    # this process already owns its GPU (pinned by setup_distributed); record it
    if torch.cuda.is_available():
        cfg.device = torch.device("cuda", torch.cuda.current_device())

    # identical base seed on every rank -> identical weight init + a consistent
    # DistributedSampler partition. DDP also broadcasts rank-0 weights at wrap.
    set_seed(cfg.seed, cfg.deterministic)

    # only rank 0 does disk I/O; other ranks get a no-op logger
    logger = DirectoryTrainingLogger(cfg) if is_main_process() else NullLogger()

    model   = BasicMobileNetV3(output_classes = cfg.model_output_class)
    wrapper = WrapperForClassification(model = model, configuration = cfg, logger = logger)

    # ---- training data: sharded across ranks under DDP ----
    train_ds  = DirectoryClassification(cfg.dataset_root_train, augmentation = heavy_aug(), seed = cfg.seed)
    train_gen = torch.Generator().manual_seed(cfg.seed)   # reproducible shuffling
    train_loader = train_ds.create_dataloader(
        batch_size    = cfg.batch_size,
        total_workers = cfg.total_workers,
        device        = cfg.device,
        shuffle       = True,
        generator     = train_gen,
        distributed   = (world_size > 1),
        seed          = cfg.seed,
    )

    # ---- eval data: rank 0 only, plain loader, no shuffle ----
    eval_loader = None
    if is_main_process():
        val_ds = DirectoryClassification(cfg.dataset_root_val, seed = cfg.seed)
        eval_loader = val_ds.create_dataloader(
            batch_size    = cfg.batch_size,
            total_workers = cfg.total_workers,
            device        = cfg.device,
            shuffle       = False,
        )

    wrapper.run_everything(
        train_dataloader = train_loader,
        test_dataloader  = eval_loader,
        test_every       = cfg.test_every_n,
        save_every       = cfg.save_every_n,
    )

    # ---- final report on the best checkpoint: rank 0 only ----
    if is_main_process():
        test_ds     = DirectoryClassification(cfg.dataset_root_test, seed = cfg.seed)
        test_loader = test_ds.create_dataloader(batch_size = cfg.batch_size, shuffle = False)
        wrapper.load_state(logger.get_weights_path("best.pt"))
        wrapper.test_model(
            test_dataloader = test_loader,
            report_prefix   = "Best",
            class_names     = test_ds.class_names,
        )


if __name__ == "__main__":
    print("Train Start !")
    configuration = build_config()
    launch(
        worker,
        configuration,
        backend         = configuration.ddp_backend,
        timeout_seconds = configuration.ddp_timeout_seconds,
    )
