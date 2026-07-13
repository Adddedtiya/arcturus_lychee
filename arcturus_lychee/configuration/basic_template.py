import torch
from dataclasses import dataclass, field

def _default_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
 
def _default_dtype() -> torch.dtype:
    # bfloat16 on capable CUDA GPUs, float32 everywhere else (incl. CPU)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


@dataclass
class TrainingConfiguration:

    # automatic configuration
    device : torch.device = field(default_factory = _default_device)
    dtype  : torch.dtype  = field(default_factory = _default_dtype)


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
    batch_size    : int = 8
    total_workers : int = 4
    test_every_n  : int = 4
    save_every_n  : int = 8

    # generic training hyperparameters
    learning_rate : float = 1e-5

    # seed management
    seed          : int  = 42
    deterministic : bool = False
    
    # model configuration should be defined somewhere else...
    
    