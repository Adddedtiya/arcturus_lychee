import torch

from arcturus_lychee.configuration                         import TrainingConfiguration
from arcturus_lychee.helpers.training_logging              import DirectoryTrainingLogger
from arcturus_lychee.datasets.basic_classification_dataset import DirectoryClassification
from arcturus_lychee.trainers.basic_classification         import WrapperForClassification
from arcturus_lychee.models.architecture.mobile_net        import BasicMobileNetV3
from arcturus_lychee.datasets                              import heavy_aug
from arcturus_lychee.helpers                               import set_seed
def generic_model_training():
    
    # setup paramters
    configuration = TrainingConfiguration()

    # setup the variables
    configuration.working_directory = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\results"
    configuration.experiment_name   = "basic_plants_training"

    # dataset
    configuration.dataset_root_train = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\plant_classification\\train"
    configuration.dataset_root_val   = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\plant_classification\\val"
    configuration.dataset_root_test  = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\plant_classification\\test"

    # training configuration
    configuration.total_epochs = 8
    configuration.batch_size   = 32

    # model configuration
    configuration.model_output_class = 40

    # seed everything BEFORE building the model / datasets / dataloaders so
    # weight init and shuffling are reproducible. Use a seeded generator for the
    # training loader so shuffling + (worker) augmentation are reproducible too.
    set_seed(configuration.seed, configuration.deterministic)
    data_generator = torch.Generator().manual_seed(configuration.seed)

    # setup the logger first
    logger = DirectoryTrainingLogger(configuration)

    # setup the model and device
    model = BasicMobileNetV3(output_classes = configuration.model_output_class)

    # create the wrapper
    wrapper = WrapperForClassification(
        model        = model,
        logger       = logger
    )

    # create the dataset
    training_dataset = DirectoryClassification(
        root_dir_path = configuration.dataset_root_train,
        augmentation  = heavy_aug(),
        seed          = configuration.seed
    )
    validation_dataset = DirectoryClassification(
        root_dir_path = configuration.dataset_root_val,
        seed          = configuration.seed
    )
    testing_dataset = DirectoryClassification(
        root_dir_path = configuration.dataset_root_test,
        seed          = configuration.seed
    )

    # create the main dataloades
    train_dataloader = training_dataset.create_dataloader(
        batch_size    = configuration.batch_size,
        total_workers = configuration.total_workers,
        device        = configuration.device,
        generator     = data_generator
    )
    eval_dataloader = validation_dataset.create_dataloader(
        batch_size    = configuration.batch_size,
        total_workers = configuration.total_workers,
        device        = configuration.device,
    )

    # start training...
    wrapper.run_everything(
        train_dataloader = train_dataloader,
        test_dataloader  = eval_dataloader,
        test_every       = configuration.test_every_n,
        save_every       = configuration.save_every_n
    )

    # training is finished...
    test_loader = testing_dataset.create_dataloader(
        batch_size = configuration.batch_size
    )
    wrapper.test_model(
        test_dataloader = test_loader,
        report_prefix   = 'Last'
    )

    # test for the best epoch
    wrapper.load_state(logger.get_weights_path("best.pt"))
    wrapper.test_model(
        test_dataloader = test_loader,
        report_prefix   = 'Best',
        class_names     = testing_dataset.class_names
    )

    # Done...

if __name__ == "__main__":
    print("Train Start !")

    generic_model_training()