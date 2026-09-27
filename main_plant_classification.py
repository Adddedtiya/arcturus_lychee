"""An example fork of main.py: plant classification with 40 classes.

Start it with: python main_plant_classification.py
The launch is the same as in main.py. The script uses one GPU or more GPUs.
This file shows the values of one project on top of the template: the
number of classes, the dataset directories, and the heavy augmentations.
It also makes two reports: one with the last weights, and one with best.pt.
"""

# Each process imports this file again. As a result:
#   - A change to a file-level variable does not get to the other processes.
#   - Code at file level operates one time in each process.
# Put all values in build_config(). launch() gives the configuration to each process.

import albumentations as A

from arcturus_lychee.configuration                  import TrainingConfiguration
from arcturus_lychee.helpers                        import (
    launch, set_seed, is_main_process,
    DirectoryTrainingLogger, NullLogger,
    build_dataloader, describe_environment,
)
from arcturus_lychee.datasets                       import DirectoryClassification, heavy_aug
from arcturus_lychee.trainers.basic_classification  import ClassificationTrainer
from arcturus_lychee.models.architecture.mobile_net import BasicMobileNetV3


def build_config() -> TrainingConfiguration:
    """Make the configuration of this training. Do not use CUDA here."""
    configuration = TrainingConfiguration()

    configuration.working_directory = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\results"
    configuration.experiment_name   = "basic_plants_training"

    configuration.dataset_root_train = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\plant_classification\\train"
    configuration.dataset_root_val   = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\plant_classification\\val"
    configuration.dataset_root_test  = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\plant_classification\\test"

    configuration.total_epochs       = 4
    configuration.test_every_n       = 2
    configuration.batch_size         = 4     # For each GPU
    configuration.model_output_class = 40
    return configuration


def worker(rank : int, world_size : int, configuration : TrainingConfiguration) -> None:
    """Train on one GPU. launch() calls this function one time in each process."""

    set_seed(configuration.seed, configuration.deterministic)

    logger      = DirectoryTrainingLogger(configuration) if is_main_process() else NullLogger()
    environment = describe_environment(configuration, world_size)
    logger.log(environment)
    logger.record.save_text("environment.txt", environment)

    model   = BasicMobileNetV3(output_classes = configuration.model_output_class)
    trainer = ClassificationTrainer(model, configuration, logger)

    training_dataset = DirectoryClassification(
        configuration.dataset_root_train,
        augmentation = heavy_aug(),
        training     = True,
        seed         = configuration.seed,
    )
    train_dataloader = build_dataloader(
        training_dataset,
        batch_size    = configuration.batch_size,
        total_workers = configuration.total_workers,
        shuffle       = True,
        distributed   = world_size > 1,
        seed          = configuration.seed,
    )
    logger.record.save_json("augmentation_train.json", A.to_dict(training_dataset.augmentation))
    logger.record.save_text("class_names.txt", training_dataset.class_names)

    eval_dataloader = None
    if is_main_process():
        validation_dataset = DirectoryClassification(configuration.dataset_root_val, training = False)
        eval_dataloader    = build_dataloader(
            validation_dataset,
            batch_size    = configuration.batch_size,
            total_workers = configuration.total_workers,
            shuffle       = False,
        )
        logger.record.save_json("augmentation_eval.json", A.to_dict(validation_dataset.augmentation))

    trainer.fit(train_dataloader, eval_dataloader)

    if is_main_process():
        testing_dataset = DirectoryClassification(configuration.dataset_root_test, training = False)
        test_loader     = build_dataloader(
            testing_dataset,
            batch_size    = configuration.batch_size,
            total_workers = configuration.total_workers,
            shuffle       = False,
        )

        # A report with the last weights, then a report with best.pt
        trainer.report(test_loader, "Last", class_names = testing_dataset.class_names)
        trainer.load_state(logger.get_weights_path("best.pt"))
        trainer.report(test_loader, "Best", class_names = testing_dataset.class_names)


if __name__ == "__main__":
    configuration = build_config()
    launch(
        worker,
        configuration,
        backend         = configuration.ddp_backend,
        timeout_seconds = configuration.ddp_timeout_seconds,
    )
