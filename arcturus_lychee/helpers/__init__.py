# Shared tools of the template. A project uses these tools without a change.
# This file is the table of contents. Import the tools from here.

# Logs, plots, and CSV files
from arcturus_lychee.helpers.training_logging import (
    DirectoryTrainingLogger,
    NullLogger,
)

# The record/ directory of a run: configuration, code, model, and more
from arcturus_lychee.helpers.experiment_record import (
    ExperimentRecord,
    NullRecord,
)

# Timer for the epochs
from arcturus_lychee.helpers.speedster_tracker import (
    SpeedTimer
)

# Image files in a directory
from arcturus_lychee.helpers.image_directory import (
    scan_directory_for_images
)

# Text reports for classification
from arcturus_lychee.helpers.classification_metrics_display import (
    generate_confusion_matrix,
    generate_report
)

# Seeds for repeatable results
from arcturus_lychee.helpers.reproducibility import (
    set_seed,
    seed_worker
)

# Processes, GPUs, and reductions across ranks
from arcturus_lychee.helpers.distributed import (
    launch,
    setup_distributed,
    cleanup_distributed,
    is_dist_initialized,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    get_device,
    all_reduce_sum_,
    all_reduce_max,
)

# Data loaders and loader timing
from arcturus_lychee.helpers.data_loading import (
    build_dataloader,
    set_sampler_epoch,
    move_to_device,
    TimedLoader,
)

# Mean of metrics across batches and ranks
from arcturus_lychee.helpers.metrics import (
    MetricAccumulator,
)

# Model on its device, DDP, and precision
from arcturus_lychee.helpers.model_setup import (
    wrap_model,
    unwrap_model,
    Precision,
)

# Report about the hardware and the software
from arcturus_lychee.helpers.environment_report import (
    describe_environment,
)
