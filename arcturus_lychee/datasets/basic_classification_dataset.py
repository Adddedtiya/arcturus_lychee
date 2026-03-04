import os
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from torch.utils.data import DataLoader, Dataset
from typing import Union
from arcturus_lychee.helpers import scan_directory_for_images

class DirectoryClassification(Dataset):
    def __init__(self, root_dir_path : str, augmentation : Union[list, None] = None):
        super().__init__()

        # dataset root
        self.root_dir = root_dir_path

        # dataset holder, i.e the actual dataset
        self.dataset_list : list[tuple[str, int]] = []

        # get the classes
        self.class_names = self.__scan_for_folders()
        self.total_class = len(self.class_names)
        
        # loop over every directory and load the files
        for class_index, class_name in enumerate(self.class_names):
            class_files = scan_directory_for_images(
                root_dir = os.path.join(self.root_dir, class_name)
            )
            for file_path in class_files:
                self.dataset_list.append((
                    file_path, class_index 
                ))

        # Create Augmentation for processing (default is just resize)
        augmentation_stack  = [ transforms.Resize((256, 256)) ]
        augmentation_stack += augmentation if augmentation else []
        augmentation_stack += [
            # in the end we crop to size
            transforms.RandomCrop((224, 244)), 
            transforms.ToTensor(), 
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std  = [0.229, 0.224, 0.225]
            )
        ]

        # create the augmenatation object
        self.augmentation = transforms.Compose(augmentation_stack)

        # all files have been loaded, you may begin !

    def __scan_for_folders(self) -> list[str]:

        folder_names = os.listdir(self.root_dir)

        # ensure that only directories is in the names
        folder_names = [
            x for x in folder_names 
            if os.path.isdir(
                os.path.join(self.root_dir, x)
            )
        ]

        # sort the files and return
        folder_names.sort()
        return folder_names
    
     # return the size of the dataset
    def __len__(self) -> int:
        return len(self.dataset_list)

    # grab one item form the dataset
    def __getitem__(self, index: int):

        # grab the file and its class
        file_path, class_index = self.dataset_list[index]

        # load image into numpy RGB numpy array in pytorch format
        image = Image.open(file_path)
        image = self.augmentation(image)

        # cast to label
        label = torch.tensor(class_index)

        # Return the image [C, H, W] and label [1, ] 
        return image, label
    
    def create_dataloader(
            self, 
            batch_size    : int = 1, 
            total_workers : int = 0, 
            device        : torch.device = 'cpu', 
            shuffle       : bool = True, 
            persistent    : bool = True
        ) -> DataLoader:

        # setup states
        currently_using_gpu = (str(device).startswith('cuda')) # for [cuda, cuda:0, cuda:N, ...]
        persistent_workers  = False if total_workers == 0 else persistent 

        # create the dataloader object
        data = DataLoader(
            self, 
            batch_size         = batch_size,
            shuffle            = shuffle,
            pin_memory         = currently_using_gpu,
            num_workers        = total_workers,
            persistent_workers = persistent_workers
        )
        return data
    
if __name__ == "__main__":
    print("Basic Dataset Classification")

    dataset = DirectoryClassification(
        root_dir_path = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\LeafDataset\\validation",
        augmentation  = [
            transforms.Resize((256, 256))
        ] 
    )

    dataset_loader = dataset.create_dataloader(1, 0, 'cpu')
    for image, label in dataset_loader:
        print(image.shape)
        print(label.shape)
        print(label)
        break
    
    
