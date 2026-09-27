import os
import random

import numpy as np
import torch
from torch.utils.data import get_worker_info


def set_seed(seed : int = 42, deterministic : bool = False) -> None:
    """Set the seeds of Python, NumPy, and PyTorch (CPU and CUDA).

    Call this function at the start of the worker function. Call it before
    you make the model, the datasets, and the loaders. Then the start weights
    and the shuffle are repeatable.

    If deterministic is True, cuDNN and PyTorch also use deterministic
    algorithms. The training is then slower. A few operations have no
    deterministic version. For these operations, PyTorch gives a warning
    and does not stop.

    PYTHONHASHSEED only has an effect on processes that start after this call.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        try:
            torch.use_deterministic_algorithms(True, warn_only = True)
        except Exception:
            # An old torch version does not have this function.
            pass


def seed_worker(worker_id : int) -> None:
    """Set the seeds of one loader worker. build_dataloader() uses this function as worker_init_fn.

    PyTorch gives each loader worker a different torch seed. The seed comes
    from the generator of the DataLoader. But PyTorch does not set the seeds
    of NumPy and random. Also, albumentations 2.x has its own random generator
    in each Compose. This function sets all three from the torch seed of the
    loader worker. The seeds are different in each loader worker, and
    repeatable from run to run, because build_dataloader() gives a seeded
    generator.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # The albumentations Compose of the dataset in this loader worker
    info = get_worker_info()
    if info is not None:
        transform = getattr(info.dataset, "augmentation", None)
        if transform is not None and hasattr(transform, "set_random_seed"):
            transform.set_random_seed(int(worker_seed))
