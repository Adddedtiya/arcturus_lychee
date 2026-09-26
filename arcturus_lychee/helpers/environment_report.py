"""A report about the hardware and the software of one training.

The report helps to find the cause when the training is slow. It does not
print and does not write files. It only returns text lines.
"""

import os
import sys
import shutil
import platform
from typing import Any, Callable, Optional

import torch

from arcturus_lychee.configuration import TrainingConfiguration


# Environment variables that the report shows, if they are set.
_REPORTED_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "SINGULARITY_CONTAINER",
    "APPTAINER_CONTAINER",
    "SLURM_JOB_ID",
    "SLURM_CPUS_PER_TASK",
    "SLURM_GPUS_ON_NODE",
)


def _safe(function : Callable[[], Any], default : Any = "unknown") -> Any:
    """Return the result of the function. If the function raises an error, return the default."""
    try:
        return function()
    except Exception:
        return default


def _usable_cores() -> Optional[int]:
    """Return the number of CPU cores that this process can use, or None if it is not known."""
    try:
        # This value obeys the cgroup and SLURM limits.
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # os.sched_getaffinity does not exist on Windows and macOS.
        return None


def describe_environment(configuration : TrainingConfiguration, world_size : int) -> list[str]:
    """Return text lines about the hardware, the software, and the loader.

    Call this function in the worker function, after the process has its GPU.
    If a value is not available, the line shows "unknown".
    """
    lines = ["Environment:"]

    # Software
    lines.append(f"  Python              : {platform.python_version()}")
    lines.append(f"  torch               : {torch.__version__}")
    lines.append(f"  CUDA / cuDNN        : {torch.version.cuda or 'none'} / "
                 f"{_safe(torch.backends.cudnn.version) or 'none'}")

    # GPUs and processes
    total_gpus = torch.cuda.device_count()
    lines.append(f"  Processes           : {world_size}")
    lines.append(f"  Visible GPUs        : {total_gpus}")
    for index in range(total_gpus):
        lines.append(f"    GPU {index}             : {_safe(lambda: torch.cuda.get_device_name(index))}")

    # CPU
    usable_cores = _usable_cores()
    lines.append(f"  Usable CPU cores    : {usable_cores if usable_cores is not None else 'unknown'}")
    lines.append(f"  All CPU cores       : {os.cpu_count()}")
    lines.append(f"  OMP_NUM_THREADS     : {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    lines.append(f"  torch threads       : {torch.get_num_threads()}")

    # OpenCV makes its own threads. The report shows them only if the project uses OpenCV.
    if "cv2" in sys.modules:
        lines.append(f"  OpenCV threads      : {_safe(lambda: sys.modules['cv2'].getNumThreads())}")

    # Loader workers
    workers = getattr(configuration, "total_workers", None)
    if workers is not None:
        lines.append(f"  Loader workers      : {workers} for each process, {workers * world_size} in total")

    # Shared memory. Loader workers and NCCL use /dev/shm.
    if os.path.isdir("/dev/shm"):
        usage = _safe(lambda: shutil.disk_usage("/dev/shm"), default = None)
        if usage is not None:
            lines.append(f"  /dev/shm            : {usage.total / 1e9:.1f} GB total, {usage.free / 1e9:.1f} GB free")

    # Container and scheduler
    for name in _REPORTED_VARIABLES:
        if name in os.environ:
            lines.append(f"  {name:<20}: {os.environ[name]}")

    # Note for too many processes and loader workers on too few cores.
    if workers is not None and usable_cores is not None:
        demand = world_size * (workers + 1)
        if demand > usable_cores:
            lines.append(
                f"  NOTE: The processes and loader workers ({demand}) are more than "
                f"the usable CPU cores ({usable_cores}). The loader can be slow."
            )

    return lines
