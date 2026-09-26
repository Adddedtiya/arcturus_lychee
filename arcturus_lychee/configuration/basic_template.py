"""The configuration of one training.

This module does not import torch. Thus the parent process can make a
configuration before mp.spawn, and no CUDA context occurs on GPU 0.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfiguration:
    """All values of one training.

    Change the values in build_config() of the entry script. To add a project
    value, set a new attribute, for example configuration.dataset_root_train.
    save_config() also writes these new attributes to configuration.toml.

    Put only plain values here: numbers, strings, booleans, lists, and dicts.
    TOML can hold these values, and pickle can send them to each process.
    """

    # General
    working_directory : str  = "results"            # The directory for all run directories.
    experiment_name   : str  = "generic_training"   # The name of the run, without the date.
    dataset_root      : str  = "dataset_path"
    prefix_date       : bool = True                 # If True, the run name starts with the date and the time.

    # Best checkpoint
    metric_to_track  : str  = "top-1"   # The evaluation metric for best.pt.
    higher_is_better : bool = True

    # Epochs and loaders
    total_epochs  : int = 128
    batch_size    : int = 8    # For each GPU. The global batch is batch_size x the number of GPUs.
    total_workers : int = 4    # Loader workers for each process.
    test_every_n  : int = 4    # Evaluate after each n epochs, and after the last epoch.
    save_every_n  : int = 8    # Save latest.pt after each n epochs, and after the last epoch.

    # Hyperparameters
    learning_rate : float = 1e-5

    # Precision: "auto", "bf16", "fp16", or "fp32".
    # "auto" selects bf16 if the GPU supports it, else fp32. On the CPU, the precision is always fp32.
    precision : str = "auto"

    # Distributed training on one machine
    ddp_backend            : str  = "nccl"   # Use "gloo" for a debug on the CPU.
    ddp_timeout_seconds    : int  = 1800     # This time must be longer than the evaluation on rank 0.
    scale_lr_by_world_size : bool = False    # If True, the trainer multiplies learning_rate by the number of GPUs.
    use_sync_batchnorm     : bool = False    # This is useful if the batch on each GPU is small.
    find_unused_parameters : bool = False    # Set True only if a part of the model gets no gradient.

    # Timing
    time_with_cuda_sync : bool = True   # If True, the step times in train.csv are accurate. The cost is small.

    # Seeds
    seed          : int  = 42
    deterministic : bool = False   # If True, cuDNN uses deterministic algorithms. The training is slower.
