import torch
from dataclasses import dataclass

@dataclass
class TrainingConfiguration:

    # automatic configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    # General Configuration
    working_directory : str  = "results"
    experiment_name   : str  = "generic_training"
    dataset_root      : str  = "dataset_path"
    append_date       : bool = True
    
    # tracking
    metric_to_track  : str  = "top-1"
    higher_is_better : bool = True

    # experimental values
    total_epochs  : int = 128
    batch_size    : int = 8
    total_workers : int = 4
    test_every_n  : int = 4
    save_every_n  : int = 8

    # model configuration should be defined somewhere else...
    
    