"""Concrete example fork: 40-class plant classification.

Same portable launch model as main_train_ddp.py - `python3 main_plant_classification.py`
runs single- or multi-GPU automatically. This file exists to show the classifier
specifics (class count, dataset layout, heavy augmentation) layered on the
generic template.
"""

import torch

from arcturus_lychee.configuration                         import TrainingConfiguration
from arcturus_lychee.helpers                               import (
    launch, set_seed, is_main_process,
    DirectoryTrainingLogger, NullLogger,
)
from arcturus_lychee.datasets                              import DirectoryClassification, heavy_aug
from arcturus_lychee.trainers.basic_classification         import WrapperForClassification
from arcturus_lychee.models.architecture.mobile_net        import BasicMobileNetV3


def build_config() -> TrainingConfiguration:
    cfg = TrainingConfiguration()

    cfg.working_directory = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\results"
    cfg.experiment_name   = "basic_plants_training"

    cfg.dataset_root_train = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\plant_classification\\train"
    cfg.dataset_root_val   = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\plant_classification\\val"
    cfg.dataset_root_test  = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\plant_classification\\test"

    cfg.total_epochs       = 4
    cfg.test_every_n       = 2
    cfg.batch_size         = 4     # PER-GPU
    cfg.model_output_class = 40
    return cfg


def worker(rank: int, world_size: int, cfg: TrainingConfiguration) -> None:

    if torch.cuda.is_available():
        cfg.device = torch.device("cuda", torch.cuda.current_device())

    set_seed(cfg.seed, cfg.deterministic)

    logger  = DirectoryTrainingLogger(cfg) if is_main_process() else NullLogger()
    model   = BasicMobileNetV3(output_classes = cfg.model_output_class)
    wrapper = WrapperForClassification(model = model, configuration = cfg, logger = logger)

    training_dataset = DirectoryClassification(cfg.dataset_root_train, augmentation = heavy_aug(), seed = cfg.seed)
    train_gen        = torch.Generator().manual_seed(cfg.seed)
    train_dataloader = training_dataset.create_dataloader(
        batch_size    = cfg.batch_size,
        total_workers = cfg.total_workers,
        device        = cfg.device,
        shuffle       = True,
        generator     = train_gen,
        distributed   = (world_size > 1),
        seed          = cfg.seed,
    )

    eval_dataloader = None
    if is_main_process():
        validation_dataset = DirectoryClassification(cfg.dataset_root_val, seed = cfg.seed)
        eval_dataloader = validation_dataset.create_dataloader(
            batch_size    = cfg.batch_size,
            total_workers = cfg.total_workers,
            device        = cfg.device,
            shuffle       = False,
        )

    wrapper.run_everything(
        train_dataloader = train_dataloader,
        test_dataloader  = eval_dataloader,
        test_every       = cfg.test_every_n,
        save_every       = cfg.save_every_n,
    )

    if is_main_process():
        testing_dataset = DirectoryClassification(cfg.dataset_root_test, seed = cfg.seed)
        test_loader     = testing_dataset.create_dataloader(batch_size = cfg.batch_size, shuffle = False)

        # last-state report, then best-checkpoint report
        wrapper.test_model(test_loader, report_prefix = "Last")
        wrapper.load_state(logger.get_weights_path("best.pt"))
        wrapper.test_model(test_loader, report_prefix = "Best", class_names = testing_dataset.class_names)


if __name__ == "__main__":
    print("Train Start !")
    configuration = build_config()
    launch(
        worker,
        configuration,
        backend         = configuration.ddp_backend,
        timeout_seconds = configuration.ddp_timeout_seconds,
    )
