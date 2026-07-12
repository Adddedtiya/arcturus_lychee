import os
import random

import numpy as np
import torch
from torch.utils.data import get_worker_info


def set_seed(seed : int = 42, deterministic : bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) RNGs for reproducible runs.

    Call this as early as possible - BEFORE building the model, datasets and
    dataloaders - so weight initialisation and data shuffling are reproducible.

    Args:
        seed: the base seed.
        deterministic: if True, also force deterministic cuDNN / algorithms.
            This can slow training down and a few ops have no deterministic
            implementation (they will warn rather than error), so it is off by
            default and best reserved for exact reproducibility runs.
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
            # older torch, or an op without a deterministic kernel - ignore
            pass


def seed_worker(worker_id : int) -> None:
    """DataLoader ``worker_init_fn`` that makes per-worker randomness reproducible.

    PyTorch already gives each worker a distinct *torch* seed derived from the
    DataLoader's ``generator``, but it does NOT reseed NumPy or ``random``, and
    albumentations 2.x keeps its OWN RNG per ``Compose`` that global seeding
    does not touch. This reseeds all three from the per-worker torch seed, which
    is distinct across workers yet reproducible run-to-run when the DataLoader
    is given a seeded ``generator`` (see ``set_seed`` / ``create_dataloader``).
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # reseed the dataset's albumentations Compose for this worker
    info = get_worker_info()
    if info is not None:
        transform = getattr(info.dataset, "augmentation", None)
        if transform is not None and hasattr(transform, "set_random_seed"):
            transform.set_random_seed(int(worker_seed))
