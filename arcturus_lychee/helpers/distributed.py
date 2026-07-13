"""Distributed (multi-GPU, single-machine) utilities for arcturus_lychee.

Design goals
------------
* **Launcher-free portability.** The entrypoint is a plain script run as
  ``python3 main_train_xxx.py`` - no ``torchrun``, no ``python -m``. This works
  identically under SLURM, Docker, or a bare shell, which matters when the same
  template is forked across heterogeneous HPC systems.
* **One code path.** Every primitive here degrades to a safe no-op when the
  process group is not initialised, so the single-GPU / CPU baseline runs
  through exactly the same trainer code as the multi-GPU path.

``launch()`` inspects the environment and dispatches one of three ways:

  1. An outer launcher already spawned the processes (``RANK`` / ``WORLD_SIZE``
     present in the environment) -> initialise from the environment, run one rank.
  2. Two or more visible CUDA GPUs -> ``mp.spawn`` one worker per GPU.
  3. One GPU or CPU -> run the worker inline, with no process group at all.

The worker signature is ``worker_fn(rank: int, world_size: int, *args)``; the
process group is set up and torn down around it by ``launch()``.
"""

import os
import socket
import datetime
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# --------------------------------------------------------------------------- #
# Introspection - safe whether or not a process group exists
# --------------------------------------------------------------------------- #

def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    """Synchronise all ranks. No-op when not running distributed."""
    if is_dist_initialized():
        dist.barrier()


# --------------------------------------------------------------------------- #
# Process-group lifecycle
# --------------------------------------------------------------------------- #

def setup_distributed(
        rank            : int,
        world_size      : int,
        backend         : str = "nccl",
        timeout_seconds : int = 1800,
    ) -> None:
    """Pin this process to its GPU and initialise the process group.

    For ``world_size <= 1`` this only selects the current CUDA device (if any)
    and returns without creating a process group, so downstream code stays on
    the single-process path.
    """
    if world_size <= 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return

    # Pin the device BEFORE init_process_group so NCCL binds to the right GPU.
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Bind the process group to this rank's device. Besides silencing the
    # "barrier(): using the device under current context" warning, this lets
    # collectives (barrier, etc.) target the correct GPU explicitly and enables
    # eager NCCL initialisation instead of lazy per-collective inference.
    device_id = torch.device("cuda", rank) if torch.cuda.is_available() else None

    # rank / world_size are passed explicitly, so only MASTER_ADDR / MASTER_PORT
    # need to be in the environment (set by launch() before spawning).
    dist.init_process_group(
        backend    = backend,
        rank       = rank,
        world_size = world_size,
        timeout    = datetime.timedelta(seconds = timeout_seconds),
        device_id  = device_id,
    )
    dist.barrier()


def cleanup_distributed() -> None:
    """Tear down the process group if one exists.

    Deliberately does NOT barrier first: rank 0 may legitimately run longer than
    the others (e.g. a final rank-0-only test pass that uses no collectives), and
    a barrier here would make the other ranks wait under the NCCL timeout for no
    reason. Each rank simply destroys its own side once it is done.
    """
    if is_dist_initialized():
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Collective reductions
# --------------------------------------------------------------------------- #

def all_reduce_metric_sums(sums: dict, counts: dict) -> tuple[dict, dict]:
    """Sum per-rank weighted metric totals across all ranks.

    Given this rank's ``{metric: weighted_sum}`` and ``{metric: weight_total}``,
    return the globally reduced versions so the caller can divide to get the true
    dataset-wide (rather than per-shard) mean. No-op when not distributed.

    All ranks must call this with the *same* set of keys. Under
    ``DistributedSampler`` every rank sees an equal-length shard and runs the
    same model, so the metric key sets match by construction.
    """
    if not is_dist_initialized():
        return sums, counts

    keys = sorted(sums.keys())
    if not keys:
        return sums, counts

    device = torch.device("cuda", torch.cuda.current_device()) \
        if torch.cuda.is_available() else torch.device("cpu")

    payload = torch.tensor(
        [sums[k] for k in keys] + [counts[k] for k in keys],
        dtype  = torch.float64,
        device = device,
    )
    dist.all_reduce(payload, op = dist.ReduceOp.SUM)

    n = len(keys)
    reduced_sums   = {k: float(payload[i].item())     for i, k in enumerate(keys)}
    reduced_counts = {k: float(payload[n + i].item()) for i, k in enumerate(keys)}
    return reduced_sums, reduced_counts


# --------------------------------------------------------------------------- #
# Launcher
# --------------------------------------------------------------------------- #

def _find_free_port() -> int:
    """Ask the OS for an unused TCP port (avoids clashes on shared nodes)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _configure_omp_threads(world_size: int) -> None:
    """Prevent CPU oversubscription across the spawned processes.

    A launcher like torchrun sets OMP_NUM_THREADS=1 for us; running bare, each of
    the N processes would otherwise grab every core. We split the affinity-aware
    core count across the processes. Respects an already-set value.
    """
    if "OMP_NUM_THREADS" in os.environ:
        return
    try:
        cores = len(os.sched_getaffinity(0))   # honours cgroup / SLURM affinity
    except AttributeError:                      # not available on all platforms
        cores = os.cpu_count() or 1
    os.environ["OMP_NUM_THREADS"] = str(max(1, cores // max(1, world_size)))


def _shutdown_reusable_executors() -> None:
    """Best-effort shutdown of joblib/loky's reusable process pool.

    scikit-learn (used for the end-of-run classification report) pulls in
    joblib, whose loky backend keeps a pool of worker processes alive after the
    call returns. If that pool outlives the process, Python's multiprocessing
    resource_tracker double-cleans its semaphores at interpreter shutdown and
    prints a noisy - but harmless - burst of "leaked semaphore" /
    ``sem_unlink FileNotFoundError`` warnings. Shutting it down here, while the
    owning process is still alive, lets the semaphores be reclaimed cleanly.

    Guarded broadly: if joblib/loky is absent or the API differs, this is a
    silent no-op and training is unaffected.
    """
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass


def _entry(rank, world_size, backend, timeout_seconds, worker_fn, worker_args) -> None:
    """Set up the group, run the worker, and always tear things down."""
    try:
        setup_distributed(rank, world_size, backend = backend, timeout_seconds = timeout_seconds)
        worker_fn(rank, world_size, *worker_args)
    finally:
        # reclaim joblib/loky worker pools before the process exits, then drop
        # the process group (order matters: do the pool cleanup while alive)
        _shutdown_reusable_executors()
        cleanup_distributed()


def launch(
        worker_fn       : Callable,
        *worker_args,
        backend         : str = "nccl",
        timeout_seconds : int = 1800,
    ) -> None:
    """Portable multi-/single-GPU entrypoint (see module docstring).

    ``worker_args`` are forwarded to every worker and so must be picklable when
    spawning (the ``TrainingConfiguration`` is; do NOT pass a built model - build
    it inside the worker, after the process owns its GPU).
    """
    # Case 1: an outer launcher already created the processes.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))  # single node: local == global
        _entry(local_rank, world_size, backend, timeout_seconds, worker_fn, worker_args)
        return

    # device_count() does not create a CUDA context, so this is spawn-safe.
    n_gpus = torch.cuda.device_count()

    # Case 2: single GPU or CPU -> inline, no process group, no spawn.
    if n_gpus <= 1:
        _entry(0, 1, backend, timeout_seconds, worker_fn, worker_args)
        return

    # Case 3: multiple GPUs on this node -> one process per GPU.
    world_size = n_gpus
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
    _configure_omp_threads(world_size)

    mp.spawn(
        _entry,
        args   = (world_size, backend, timeout_seconds, worker_fn, worker_args),
        nprocs = world_size,
        join   = True,
    )