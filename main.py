def main():
    from arcturus_lychee.datasets.basic_classification_dataset import DirectoryClassification, transforms
    dataset = DirectoryClassification(
        root_dir_path = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\LeafDataset\\validation",
        augmentation  = [
            transforms.Resize((256, 256))
        ] 
    )   

    print(dataset.augmentation)

    dataset_loader = dataset.create_dataloader(8, 0, 'cpu')
    for image, label in dataset_loader:
        print(image.shape)
        print(label.shape)
        print(label)
        break
    
    print(dataset.class_names)

if __name__ == "__main__":
    main()

    