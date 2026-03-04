import numpy as np

import torch
import torch.nn             as nn
import torch.nn.functional  as F

from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp.grad_scaler import GradScaler
from arcturus_lychee.helpers import DirectoryTrainingLogger, SpeedTimer, generate_report, generate_confusion_matrix
from torch.utils.data   import DataLoader
from collections import defaultdict

from typing import Union

from arcturus_lychee.experimental.fp8_anealing       import Float8CosineScheduler
from arcturus_lychee.trainers.wrapper_classification import WrapperForClassification

class WrapperForFloat8Classification(WrapperForClassification):
    def __init__(self, model, logger, total_epochs, device = 'cpu'):
        super().__init__(model, logger, total_epochs, device)

        # add support for Float8 cosine Scheduler
        self.float8_scheduler = Float8CosineScheduler(
            model = self.model,
            total_steps = total_epochs,
        )


    def run_single_epoch(self, train_dataloader, test_dataloader, current_epoch, enable_tqdm = True, test_every = 1):
        
        # run the original function
        train_stats, test_stats, total_runtime =  super().run_single_epoch(
            train_dataloader, 
            test_dataloader, 
            current_epoch, 
            enable_tqdm, 
            test_every
        )

        # step the float Scheduler
        self.float8_scheduler.step()
        train_stats['f8_rate'] = self.float8_scheduler.get_current_alpha()

        # return the function to original habititat
        return train_stats, test_stats, total_runtime
