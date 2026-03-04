import torch
import torch.nn as nn
import math

class Float8CosineScheduler:
    def __init__(self, model : nn.Module, total_steps : int , start_step = 0):
        self.model = model
        self.total_steps   = total_steps
        self.start_step    = start_step
        self.current_step  = 0 + 1 # assume that the first epoch allready passed
        self.alpha         = 0.0

    def __get_alpha(self):
        if self.current_step < self.start_step:
            return 0.0
        
        # Progress from 0 to 1 over the duration
        progress = (self.current_step - self.start_step) / (self.total_steps - self.start_step)
        progress = min(1.0, max(0.0, progress))
        
        # Cosine mapping
        return 0.5 * (1 - math.cos(math.pi * progress))

    def get_current_alpha(self) -> float:
        return float(self.alpha)

    def step(self):
        self.alpha = self.__get_alpha()
        
        # Apply the transformation logic
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    # Target grid for e4m3fnuz
                    q_weight = self.__simulate_e4m3fnuz(param.data)
                    
                    # Soft-blend the weights
                    # Weight = (1-alpha)*Original + alpha*Quantized
                    param.data.copy_((1 - self.alpha) * param.data + self.alpha * q_weight)
        
        self.current_step += 1

    def __simulate_e4m3fnuz(self, x):
        """
        Simulates the e4m3fnuz grid. 
        Note: This is a functional simulation. For actual hardware 
        casting, use torch.to(torch.float8_e4m3fnuz) in compatible PyTorch versions.
        """
        max_val = 240.0
        # Simple dynamic scaling per tensor
        scale = max_val / (x.abs().max() + 1e-8)
        
        x_scl = x * scale
        # 3-bit mantissa simulation via rounding to specific powers
        # For a true e4m3fnuz, we clamp to the finite range
        x_q = torch.clamp(torch.round(x_scl), -max_val, max_val)
        return x_q / scale