"""The entry script of a training. Start it with: python main.py

The script uses the visible GPUs:

  - Two or more GPUs: launch() starts one process for each GPU (DDP with NCCL).
  - One GPU or the CPU: the training operates in this process, without DDP.

The script does not need torchrun or ``python -m``. Thus the same command
works in SLURM, Singularity, Docker, and a shell. To use fewer GPUs, set
CUDA_VISIBLE_DEVICES, for example: CUDA_VISIBLE_DEVICES=0,1 python main.py

batch_size is the batch for each GPU. The global batch is batch_size x the
number of GPUs.
"""

# Each process imports this file again. As a result:
#   - A change to a file-level variable does not get to the other processes.
#   - Code at file level operates one time in each process.
# Put all values in build_config(). launch() gives the configuration to each process.

import albumentations as A

from arcturus_lychee.configuration                  import TrainingConfiguration, load_run_configuration
from arcturus_lychee.helpers                        import (
    launch, set_seed, is_main_process,
    DirectoryTrainingLogger, NullLogger,
    build_dataloader, describe_environment,
)
from arcturus_lychee.datasets                       import DirectoryClassification, heavy_aug
from arcturus_lychee.trainers.basic_classification  import ClassificationTrainer
from arcturus_lychee.models.architecture.mobile_net import BasicMobileNetV3


def build_config() -> TrainingConfiguration:
    """Make the configuration of this training.

    The parent process calls this function before launch() starts the other
    processes. Do not use CUDA here. Do not make models or tensors here.
    """
    configuration = TrainingConfiguration()

    # To do an ablation, load the configuration of a previous run and change one value:
    # configuration = load_run_configuration("results/2026_09_26_10_00-baseline")
    # configuration.learning_rate   = 1e-4
    # configuration.experiment_name = "baseline_lr_1e-4"

    configuration.working_directory = "results"
    configuration.experiment_name   = "ddp_training"

    # Dataset directories, with one subdirectory for each class.
    configuration.dataset_root_train = "/path/to/dataset/train"
    configuration.dataset_root_val   = "/path/to/dataset/val"
    configuration.dataset_root_test  = "/path/to/dataset/test"

    configuration.total_epochs       = 8
    configuration.batch_size         = 32      # For each GPU
    configuration.total_workers      = 4       # Loader workers for each process
    configuration.model_output_class = 40

    # Distributed training. The defaults are safe. Change them for each experiment.
    configuration.scale_lr_by_world_size = False   # True: learning_rate x the number of GPUs
    configuration.use_sync_batchnorm     = False   # True: useful if the batch on each GPU is small
    return configuration


def worker(rank : int, world_size : int, configuration : TrainingConfiguration) -> None:
    """Train on one GPU. launch() calls this function one time in each process."""

    # The same seed on each rank gives the same start weights.
    # DDP also copies the weights of rank 0 to the other ranks.
    set_seed(configuration.seed, configuration.deterministic)

    # Only rank 0 writes files. The other ranks get a logger that does nothing.
    logger      = DirectoryTrainingLogger(configuration) if is_main_process() else NullLogger()
    environment = describe_environment(configuration, world_size)
    logger.log(environment)
    logger.record.save_text("environment.txt", environment)

    model   = BasicMobileNetV3(output_classes = configuration.model_output_class)
    trainer = ClassificationTrainer(model, configuration, logger)

    # Training data. With DDP, each rank gets a different part of the dataset.
    train_set    = DirectoryClassification(
        configuration.dataset_root_train,
        augmentation = heavy_aug(),
        training     = True,
        seed         = configuration.seed,
    )
    train_loader = build_dataloader(
        train_set,
        batch_size    = configuration.batch_size,
        total_workers = configuration.total_workers,
        shuffle       = True,
        distributed   = world_size > 1,
        seed          = configuration.seed,
    )
    logger.record.save_json("augmentation_train.json", A.to_dict(train_set.augmentation))
    logger.record.save_text("class_names.txt", train_set.class_names)

    # Evaluation data: rank 0 only, without shuffle.
    eval_loader = None
    if is_main_process():
        eval_set    = DirectoryClassification(configuration.dataset_root_val, training = False)
        eval_loader = build_dataloader(
            eval_set,
            batch_size    = configuration.batch_size,
            total_workers = configuration.total_workers,
            shuffle       = False,
        )
        logger.record.save_json("augmentation_eval.json", A.to_dict(eval_set.augmentation))

    trainer.fit(train_loader, eval_loader)

    # Last report with the best checkpoint: rank 0 only.
    if is_main_process():
        test_set    = DirectoryClassification(configuration.dataset_root_test, training = False)
        test_loader = build_dataloader(
            test_set,
            batch_size    = configuration.batch_size,
            total_workers = configuration.total_workers,
            shuffle       = False,
        )
        trainer.load_state(logger.get_weights_path("best.pt"))
        trainer.report(test_loader, "Best", class_names = test_set.class_names)


if __name__ == "__main__":
    configuration = build_config()
    launch(
        worker,
        configuration,
        backend         = configuration.ddp_backend,
        timeout_seconds = configuration.ddp_timeout_seconds,
    )
