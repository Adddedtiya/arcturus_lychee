import torch
from arcturus_lychee.helpers.training_logging              import DirectoryTrainingLogger
from arcturus_lychee.datasets.basic_classification_dataset import DirectoryClassification, transforms
from arcturus_lychee.trainers.wrapper_classification       import WrapperForClassification
from arcturus_lychee.models.architecture.mobile_net        import BasicMobileNetV3


def train_plants():
    
    # general paramters
    device        = torch.device('cuda')
    total_epochs  = 2
    batch_size    = 64
    total_workers = 8
    test_every_n  = 4
    save_every_n  = 8

    # setup the logger first
    logger = DirectoryTrainingLogger(
        working_directory = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\results",
        best_metric       = 'accuracy',
        higher_is_better  = True,
        experiment_name   = 'PlanClassificaionBF16x2'
    )

    # setup the model and device
    model = BasicMobileNetV3(output_classes = 40)

    # create the wrapper
    wrapper = WrapperForClassification(
        model        = model,
        logger       = logger,
        total_epochs = total_epochs,
        device       = device
    )

    # create the dataset
    training_dataset = DirectoryClassification(
        root_dir_path = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\plant_classification\\train",
        augmentation  = [
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ] 
    )
    validation_dataset = DirectoryClassification(
        root_dir_path = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\plant_classification\\val",
    )
    testing_dataset = DirectoryClassification(
        root_dir_path = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\plant_classification\\test",
    )

    # create the main dataloades
    train_dataloader = training_dataset.create_dataloader(
        batch_size    = batch_size,
        total_workers = total_workers,
        device        = device,
    )
    eval_dataloader = validation_dataset.create_dataloader(
        batch_size    = batch_size,
        total_workers = total_workers,
        device        = device,
    )

    # start training...
    wrapper.run_everything(
        train_dataloader = train_dataloader,
        test_dataloader  = eval_dataloader,
        test_every       = test_every_n,
        save_every       = save_every_n
    )

    # training is finished...
    test_loader = testing_dataset.create_dataloader(
        batch_size = batch_size
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

    train_plants()