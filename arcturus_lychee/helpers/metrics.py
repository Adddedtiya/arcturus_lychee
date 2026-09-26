"""A task-independent tool that calculates the mean of metrics during an epoch."""

import torch
from typing import Union

from arcturus_lychee.helpers.distributed import get_device, all_reduce_sum_


class MetricAccumulator:
    """Collect weighted metric values, and calculate the mean at the end.

    The values stay on the device as tensors. Thus add() does not make the CPU
    wait for the GPU. Only reduce() waits for the GPU. The metric names are free, for
    example "loss", "top-1", "dice", or "psnr".

    Example:

        meter = MetricAccumulator()
        for batch in loader:
            ...
            meter.add("loss", loss, weight = images.size(0))
        averages = meter.reduce()        # {"loss": 0.42}
    """

    def __init__(self) -> None:
        self.sums    : dict[str, torch.Tensor] = {}
        self.weights : dict[str, float]        = {}

    def add(self, name : str, value : Union[float, torch.Tensor], weight : float = 1.0) -> None:
        """Add one value with its weight. The value is a number or a tensor with one element.

        Usually the weight is the number of samples in the batch.
        """
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"The value for '{name}' must have one element. It has {value.numel()} elements."
                )
            value = value.detach().reshape(()).double()
        else:
            value = torch.tensor(float(value), dtype = torch.float64)

        weighted = value * float(weight)
        if name in self.sums:
            self.sums[name] = self.sums[name] + weighted.to(self.sums[name].device)
        else:
            self.sums[name] = weighted
        self.weights[name] = self.weights.get(name, 0.0) + float(weight)

    def reduce(self, across_ranks : bool = True) -> dict[str, float]:
        """Return the weighted mean of each metric.

        If across_ranks is True, the mean includes the values of all ranks.
        Thus the result is the mean of the full dataset, not of one part.
        In this case, all ranks must call reduce() and must have the same names.
        For an evaluation on rank 0 only, use across_ranks=False.
        """
        keys = sorted(self.sums.keys())
        if not keys:
            return {}

        device  = get_device()
        sums    = torch.stack([self.sums[k].to(device) for k in keys])
        weights = torch.tensor([self.weights[k] for k in keys], dtype = torch.float64, device = device)
        payload = torch.cat([sums, weights])

        if across_ranks:
            all_reduce_sum_(payload)

        # One copy to the CPU. This is the only GPU sync.
        payload = payload.cpu()
        n       = len(keys)
        return {
            key: float(payload[i] / payload[n + i])
            for i, key in enumerate(keys)
            if payload[n + i] != 0
        }
