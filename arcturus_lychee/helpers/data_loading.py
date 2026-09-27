"""Shared tools for data loaders. These tools work with a dataset of any type.

The tools do not know the task or the shape of a batch. A batch can be a
tensor, a tuple, a list, or a dict, for example 2D images, 3D volumes, or audio.
"""

import time
from typing import Any, Callable, Iterator, Optional, Union

import torch
from torch.utils.data             import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from arcturus_lychee.helpers.distributed     import get_rank
from arcturus_lychee.helpers.reproducibility import seed_worker


def build_dataloader(
        dataset       : Dataset,
        batch_size    : int,
        total_workers : int  = 0,
        shuffle       : bool = True,
        distributed   : bool = False,
        seed          : int  = 0,
        drop_last     : bool = False,
        collate_fn    : Optional[Callable[[list], Any]] = None,
    ) -> DataLoader:
    """Make a DataLoader for a dataset of any type.

    If distributed is True, each rank gets a different part of the dataset.
    The DistributedSampler then controls the shuffle. The trainer calls
    set_sampler_epoch() at the start of each epoch, so the order changes.

    For the validation set and the test set, use distributed=False and
    shuffle=False. Only rank 0 does the evaluation.

    total_workers is the number of loader workers for each process.

    collate_fn makes one batch from a list of samples. Without it, PyTorch
    stacks the samples, and all samples must have the same shape. For samples
    with different lengths, for example audio, give a collate_fn that pads them.
    The collate_fn must be a function at file level, so that pickle can send it
    to the loader workers.
    """
    sampler = None
    if distributed:
        # All ranks must use the same seed. The sampler then gives each rank a different part.
        sampler = DistributedSampler(
            dataset,
            shuffle   = shuffle,
            seed      = seed,
            drop_last = drop_last,
        )

    # The generator controls the shuffle without a sampler, and the seeds of the loader workers.
    # Each rank adds its rank to the seed. Thus the random augmentations are different on each rank.
    # The result is repeatable: the same seed and the same number of GPUs give the same values.
    # With one GPU, the rank is 0, and the seed does not change.
    generator = torch.Generator().manual_seed(seed + get_rank())

    return DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = shuffle and sampler is None,
        sampler            = sampler,
        num_workers        = total_workers,
        pin_memory         = torch.cuda.is_available(),
        persistent_workers = total_workers > 0,
        worker_init_fn     = seed_worker if total_workers > 0 else None,
        generator          = generator,
        drop_last          = drop_last,
        collate_fn         = collate_fn,
    )


def set_sampler_epoch(loader : Union[DataLoader, "TimedLoader"], epoch : int) -> None:
    """Give the epoch number to the DistributedSampler of the loader, if it has one.

    Without this call, the sampler uses the same order in each epoch.
    """
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


def move_to_device(batch : Any, device : torch.device) -> Any:
    """Move all tensors in a batch to the device, and return the batch.

    The batch can be a tensor, a tuple, a list, a named tuple, or a dict.
    These types can also contain each other. Other values do not change.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking = True)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):
        return type(batch)(*(move_to_device(value, device) for value in batch))
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(value, device) for value in batch)
    return batch


class TimedLoader:
    """Wrapper for a loader that measures the time of one epoch.

    The wrapper measures two times for each batch:

      - Data wait: the time in next(). The GPU has no work in this time.
      - Step: the time from the end of next() to the next call of next().
        This is the time of the loop body, for example forward and backward.

    If the data wait is a large part of the total, the loader is too slow.
    Make one TimedLoader for each epoch. The trainer does this.
    """

    def __init__(self, loader : DataLoader, sync_cuda : bool = True) -> None:
        self.loader  = loader
        self.sampler = getattr(loader, "sampler", None)

        # CUDA operations are asynchronous. Without a sync, a part of the GPU time
        # goes into the data wait, and the data wait is too high.
        self.sync_cuda = sync_cuda and torch.cuda.is_available()

        self.data_wait_seconds : float = 0.0
        self.step_seconds      : float = 0.0

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Any]:
        iterator = iter(self.loader)
        while True:
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            received                = time.perf_counter()
            self.data_wait_seconds += received - start

            # The loop body of the caller operates during this yield.
            yield batch

            if self.sync_cuda:
                torch.cuda.synchronize()
            self.step_seconds += time.perf_counter() - received

    def stats(self) -> dict[str, float]:
        """Return the times of this epoch, in seconds and in percent."""
        total = self.data_wait_seconds + self.step_seconds
        return {
            "time_data_wait_s"   : self.data_wait_seconds,
            "time_step_s"        : self.step_seconds,
            "time_data_wait_pct" : 100.0 * self.data_wait_seconds / total if total > 0 else 0.0,
        }
