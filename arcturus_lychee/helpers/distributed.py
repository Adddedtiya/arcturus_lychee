"""Utilities for distributed training on one machine with one or more GPUs.

The entry script starts with ``python main.py``. The template does not use
torchrun or ``python -m``. As a result, the same command works in SLURM,
Singularity, Docker, and a shell.

All functions in this module also work in a single process. If no process
group exists, each function does nothing or gives the single-process value.
Thus the single-GPU path and the multi-GPU path use the same trainer code.

``launch()`` selects one of two paths:

  1. Zero GPUs or one GPU: the worker function operates in this process.
     No process group exists.
  2. Two or more GPUs: ``mp.spawn`` starts one process for each GPU.

The signature of the worker function is ``worker_fn(rank, world_size, *args)``.
``launch()`` makes the process group before the worker function starts. It
removes the process group after the worker function stops.
"""

import os
import socket
import datetime
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# --------------------------------------------------------------------------- #
# Information about the processes. These functions also work without a
# process group.
# --------------------------------------------------------------------------- #

def is_dist_initialized() -> bool:
    """Return True if a process group exists."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Return the number of processes. The value is 1 if no process group exists."""
    return dist.get_world_size() if is_dist_initialized() else 1


def get_rank() -> int:
    """Return the rank of this process. The value is 0 if no process group exists."""
    return dist.get_rank() if is_dist_initialized() else 0


def is_main_process() -> bool:
    """Return True for rank 0. Rank 0 does the evaluation, the logs, and the checkpoints."""
    return get_rank() == 0


def barrier() -> None:
    """Stop each process until all processes get to this line.

    If no process group exists, this function does nothing.
    """
    if is_dist_initialized():
        dist.barrier()


def get_device() -> torch.device:
    """Return the device of this process.

    The value is ``cuda:<n>`` if CUDA is available, else ``cpu``.
    ``setup_distributed()`` selects the GPU of each process before the worker
    function starts. Thus this function is the only source for the device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Start and stop of the process group
# --------------------------------------------------------------------------- #

def setup_distributed(
        rank            : int,
        world_size      : int,
        backend         : str = "nccl",
        timeout_seconds : int = 1800,
    ) -> None:
    """Select the GPU of this process and make the process group.

    If world_size is 1 or less, this function only selects GPU 0 (if CUDA is
    available). It does not make a process group. Thus the code after it uses
    the single-process path.
    """
    if world_size <= 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return

    # Select the GPU before init_process_group. NCCL then uses the correct GPU.
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # device_id connects the process group to the GPU of this rank. Collective
    # operations then use the correct GPU, and NCCL starts immediately.
    device_id = torch.device("cuda", rank) if torch.cuda.is_available() else None

    # The call gives rank and world_size. Thus only MASTER_ADDR and MASTER_PORT
    # must be in the environment. launch() sets these two values.
    dist.init_process_group(
        backend    = backend,
        rank       = rank,
        world_size = world_size,
        timeout    = datetime.timedelta(seconds = timeout_seconds),
        device_id  = device_id,
    )
    dist.barrier()


def cleanup_distributed() -> None:
    """Remove the process group, if it exists.

    This function does not call a barrier first. Rank 0 can operate longer
    than the other ranks, for example for the last test on rank 0 only.
    With a barrier, the other ranks wait for rank 0, and the NCCL timeout can
    stop them.
    """
    if is_dist_initialized():
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Collective reductions
# --------------------------------------------------------------------------- #

def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    """Add the tensors of all ranks, in place, and return the result.

    After the call, each rank has the sum of all ranks. If no process group
    exists, the tensor does not change. All ranks must call this function
    with a tensor of the same shape.
    """
    if is_dist_initialized():
        dist.all_reduce(tensor, op = dist.ReduceOp.SUM)
    return tensor


def all_reduce_max(values: dict[str, float]) -> dict[str, float]:
    """Return the maximum of each value across all ranks.

    The trainer uses this function for times, because the slowest rank sets
    the speed of the training. All ranks must give the same keys. If no
    process group exists, the function returns the values without a change.
    """
    if not is_dist_initialized() or not values:
        return dict(values)

    keys    = sorted(values.keys())
    payload = torch.tensor([values[k] for k in keys], dtype = torch.float64, device = get_device())
    dist.all_reduce(payload, op = dist.ReduceOp.MAX)
    return {k: float(payload[i].item()) for i, k in enumerate(keys)}


def all_reduce_metric_sums(sums: dict, counts: dict) -> tuple[dict, dict]:
    """Add the weighted metric sums and the weight totals of all ranks.

    Old API. The current trainer uses this function. The trainer rewrite
    replaces it with MetricAccumulator, and then removes this function.
    All ranks must give the same keys.
    """
    if not is_dist_initialized():
        return sums, counts

    keys = sorted(sums.keys())
    if not keys:
        return sums, counts

    payload = torch.tensor(
        [sums[k] for k in keys] + [counts[k] for k in keys],
        dtype  = torch.float64,
        device = get_device(),
    )
    all_reduce_sum_(payload)

    n = len(keys)
    reduced_sums   = {k: float(payload[i].item())     for i, k in enumerate(keys)}
    reduced_counts = {k: float(payload[n + i].item()) for i, k in enumerate(keys)}
    return reduced_sums, reduced_counts


# --------------------------------------------------------------------------- #
# Launcher
# --------------------------------------------------------------------------- #

# Singularity and Apptainer. If a problem occurs, do the applicable step:
#   - If torch.cuda.device_count() is 0 in the container, add the --nv flag
#     to the container command.
#   - If a loader worker stops with "bus error", decrease total_workers or
#     increase the size of /dev/shm.
#   - If the NCCL communication between the GPUs stops, set
#     NCCL_P2P_DISABLE=1 before you start the script.
#   - If the NCCL communication still stops, also set NCCL_IB_DISABLE=1.
# Note: the --contain and --containall flags give the container a small,
# private /dev/shm. The two NCCL values decrease the speed. As a result,
# the template does not set them.

# torchrun sets these variables. srun sets SLURM_STEP_NUM_TASKS.
_EXTERNAL_LAUNCHER_VARIABLES = ("RANK", "WORLD_SIZE", "LOCAL_RANK")


def _refuse_external_launcher() -> None:
    """Stop with an error if an external launcher started this script.

    An external launcher starts N copies of the script. If each copy then
    calls launch(), each copy starts one process for each GPU. With N GPUs,
    the result is N x N processes on N GPUs.
    """
    found = None

    for name in _EXTERNAL_LAUNCHER_VARIABLES:
        if name in os.environ:
            found = f"The environment variable {name} is set."

    step_tasks = os.environ.get("SLURM_STEP_NUM_TASKS", "1")
    if step_tasks.isdigit() and int(step_tasks) > 1:
        found = f"SLURM_STEP_NUM_TASKS is {step_tasks}."

    if found is None:
        return

    raise RuntimeError(
        f"An external launcher started this script. {found} "
        "This template starts one process for each GPU itself. "
        "Start the script one time with 'python main.py'. "
        "Do not use torchrun. Do not use srun with more than one task."
    )


def _find_free_port() -> int:
    """Get an unused TCP port from the operating system.

    A fixed port can already be in use on a shared node.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _configure_omp_threads(world_size: int) -> None:
    """Divide the usable CPU cores between the processes.

    torchrun sets OMP_NUM_THREADS=1. Without a launcher, each of the N
    processes uses all cores, and the processes compete for the cores.
    This function gives each process an equal part of the usable cores.
    If OMP_NUM_THREADS is already set, the function does not change it.
    """
    if "OMP_NUM_THREADS" in os.environ:
        return
    try:
        # The usable cores. This value obeys the cgroup and SLURM limits.
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        # os.sched_getaffinity does not exist on Windows and macOS.
        cores = os.cpu_count() or 1
    os.environ["OMP_NUM_THREADS"] = str(max(1, cores // max(1, world_size)))


def _shutdown_reusable_executors() -> None:
    """Stop the process pool of joblib (loky), if the pool exists.

    scikit-learn makes the classification report, and scikit-learn uses
    joblib. The loky backend of joblib keeps its processes alive after the
    call. At interpreter exit, the resource_tracker of multiprocessing then
    cleans the same semaphores two times. The result is a long list of
    "leaked semaphore" warnings. These warnings are harmless.

    This function stops the pool while the process is alive. Thus the
    semaphores close correctly. If joblib is not available, or its API is
    different, this function does nothing.
    """
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait = True)
    except Exception:
        pass


def _entry(rank, world_size, backend, timeout_seconds, worker_fn, worker_args) -> None:
    """Make the process group, start the worker function, and always remove the process group at the end."""
    try:
        setup_distributed(rank, world_size, backend = backend, timeout_seconds = timeout_seconds)
        worker_fn(rank, world_size, *worker_args)
    finally:
        # Stop the joblib pool first, while the process is alive.
        # Then remove the process group.
        _shutdown_reusable_executors()
        cleanup_distributed()


def launch(
        worker_fn       : Callable,
        *worker_args,
        backend         : str = "nccl",
        timeout_seconds : int = 1800,
    ) -> None:
    """Start the worker function in this process, or in one process for each GPU.

    The module docstring describes the two paths. The worker_args go to each
    process. With two or more GPUs, mp.spawn pickles the worker_args.
    A TrainingConfiguration is safe to pickle.

    Do not give a model to this function. Make the model inside the worker
    function, after the process has its GPU.
    """
    _refuse_external_launcher()

    # device_count() does not make a CUDA context. Thus the call is safe before mp.spawn.
    n_gpus = torch.cuda.device_count()

    # Path 1: zero GPUs or one GPU. The worker function operates in this process.
    if n_gpus <= 1:
        _entry(0, 1, backend, timeout_seconds, worker_fn, worker_args)
        return

    # Path 2: two or more GPUs. Start one process for each GPU.
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
