"""Shared tools to put a model on its device, wrap it for DDP, and control the precision."""

import torch
import torch.nn as nn

from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel     import DistributedDataParallel as DDP

from arcturus_lychee.helpers.distributed import get_world_size


def wrap_model(
        model                  : nn.Module,
        device                 : torch.device,
        sync_batchnorm         : bool = False,
        find_unused_parameters : bool = False,
    ) -> nn.Module:
    """Move the model to the device. With more than one process, wrap it in DDP.

    If sync_batchnorm is True and more than one process exists, the BatchNorm
    layers use the statistics of all GPUs. This is useful if the batch on
    each GPU is small.

    With one process, the function returns the model without a wrapper.
    Use unwrap_model() to get the model without DDP, for evaluation and checkpoints.
    """
    model = model.to(device)

    if get_world_size() <= 1:
        return model

    # The conversion must occur before the DDP wrap.
    if sync_batchnorm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    device_ids = [device.index] if device.type == "cuda" else None
    return DDP(
        model,
        device_ids             = device_ids,
        output_device          = device_ids[0] if device_ids else None,
        find_unused_parameters = find_unused_parameters,
    )


def unwrap_model(model : nn.Module) -> nn.Module:
    """Return the model inside a DDP wrapper, or the model itself if it has no wrapper.

    Checkpoints from the unwrapped model have no "module." prefix. Thus they
    load with and without DDP.
    """
    return model.module if isinstance(model, DDP) else model


class Precision:
    """Control the autocast and the gradient scaler for one precision.

    The permitted values of precision are "auto", "bf16", "fp16", and "fp32":

      - "auto" gives bf16 on a CUDA GPU with bf16 support, else fp32.
      - "fp16" uses a GradScaler. The other values do not need it.
      - On the CPU, the precision is always fp32.

    The name attribute gives the precision that the object uses.

    One training step:

        with self.precision.autocast():
            loss = ...
        self.precision.backward(loss)
        self.precision.step(self.optimizer)
        self.optimizer.zero_grad(set_to_none = True)

    For gradient clipping, call self.precision.scaler.unscale_(optimizer)
    before the clip. For two optimizers, call self.precision.scaler.step()
    for each optimizer, then call self.precision.scaler.update() one time.
    """

    DTYPES : dict[str, torch.dtype] = {
        "bf16" : torch.bfloat16,
        "fp16" : torch.float16,
        "fp32" : torch.float32,
    }

    def __init__(self, precision : str, device : torch.device) -> None:
        if precision != "auto" and precision not in self.DTYPES:
            raise ValueError(
                f"The precision '{precision}' is not known. "
                f"Use one of these values: auto, {', '.join(self.DTYPES)}."
            )

        on_cuda = device.type == "cuda"

        if not on_cuda:
            name = "fp32"
        elif precision == "auto":
            # This call makes a CUDA context. Thus it is only safe inside the process of the GPU.
            name = "bf16" if torch.cuda.is_bf16_supported() else "fp32"
        else:
            name = precision

        self.name        = name
        self.dtype       = self.DTYPES[name]
        self.device_type = device.type
        self.enabled     = on_cuda and name != "fp32"
        self.scaler      = GradScaler(device.type, enabled = on_cuda and name == "fp16")

    def autocast(self) -> torch.autocast:
        """Return the autocast context for the forward pass and the loss."""
        return torch.autocast(device_type = self.device_type, dtype = self.dtype, enabled = self.enabled)

    def backward(self, loss : torch.Tensor) -> None:
        """Calculate the gradients. With fp16, the scaler multiplies the loss first."""
        self.scaler.scale(loss).backward()

    def step(self, optimizer : torch.optim.Optimizer) -> None:
        """Do one optimizer step, then update the scale of the scaler."""
        self.scaler.step(optimizer)
        self.scaler.update()

    def state_dict(self) -> dict:
        """Return the state of the scaler, for a checkpoint."""
        return self.scaler.state_dict()

    def load_state_dict(self, state : dict) -> None:
        """Load the state of the scaler from a checkpoint."""
        self.scaler.load_state_dict(state)
