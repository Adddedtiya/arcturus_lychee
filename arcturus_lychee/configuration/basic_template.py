import torch
from typing import Optional
from dataclasses import dataclass, field


def _default_device() -> torch.device:
    # NOTE: torch.cuda.is_available() does NOT create a CUDA context, so this is
    # safe to evaluate in a parent process *before* torch.multiprocessing.spawn.
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def default_dtype() -> torch.dtype:
    """bfloat16 on capable CUDA GPUs, float32 everywhere else (incl. CPU).

    Resolved lazily (NOT as a dataclass default) on purpose: `is_bf16_supported()`
    can initialise a CUDA context on device 0, and doing that in the parent
    process before `mp.spawn` leaves a stray context pinned on GPU 0. Call this
    from inside the worker (the trainer does so) once the process owns its GPU.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


@dataclass
class TrainingConfiguration:

    # automatic configuration
    #   `device` is safe to resolve eagerly (no CUDA context). `dtype` is left
    #   None and resolved lazily by the trainer via `default_dtype()` so building
    #   this config in a parent process does not touch the GPU before spawn.
    device : torch.device           = field(default_factory = _default_device)
    dtype  : Optional[torch.dtype]  = None

    # General Configuration
    working_directory : str  = "results"
    experiment_name   : str  = "generic_training"
    dataset_root      : str  = "dataset_path"
    prefix_date       : bool = True

    # tracking
    metric_to_track  : str  = "top-1"
    higher_is_better : bool = True

    # experimental values
    total_epochs  : int = 128
    batch_size    : int = 8          # PER-GPU / per-process batch size under DDP
    total_workers : int = 4
    test_every_n  : int = 4
    save_every_n  : int = 8

    # generic training hyperparameters
    learning_rate : float = 1e-5

    # distributed / multi-GPU (single machine, multiple NVIDIA GPUs)
    ddp_backend            : str  = "nccl"   # "gloo" for CPU-only debugging
    ddp_timeout_seconds    : int  = 1800     # covers rank-0-only eval between epochs
    scale_lr_by_world_size : bool = False    # opt-in linear LR scaling (lr *= world_size)
    use_sync_batchnorm     : bool = False    # opt-in; recommended when per-GPU batch is small
    find_unused_parameters : bool = False    # keep False for these CNN backbones (faster)

    # seed management
    seed          : int  = 42
    deterministic : bool = False

    # model configuration should be defined somewhere else...

    def resolved_dtype(self) -> torch.dtype:
        """Return the configured dtype, resolving the lazy default if unset."""
        return self.dtype if self.dtype is not None else default_dtype()
