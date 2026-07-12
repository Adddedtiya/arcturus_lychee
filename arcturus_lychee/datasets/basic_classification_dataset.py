import os
import torch
import numpy as np

from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader, Dataset
from typing import Union
from arcturus_lychee.helpers import scan_directory_for_images, seed_worker

class DirectoryClassification(Dataset):
    def __init__(self, root_dir_path : str, augmentation : Union[list, None] = None, seed : Union[int, None] = None):
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

        # Create Augmentation for processing (default is just resize).
        # albumentations pipeline: numpy HWC uint8 in -> CHW float tensor out.
        # Normalize must come before ToTensorV2 (Normalize expects HWC; ToTensorV2
        # only transposes to CHW and does NOT rescale, so it goes last).
        augmentation_stack  = [ A.Resize(256, 256) ]
        augmentation_stack += augmentation if augmentation else []
        augmentation_stack += [
            # in the end we crop to size
            A.RandomCrop(224, 224),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std  = [0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]
 
        # create the augmenatation object
        self.augmentation = A.Compose(augmentation_stack, seed = seed)

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
 
        # load image as an RGB numpy array (HWC uint8) for albumentations.
        # .convert("RGB") guarantees 3 channels so grayscale / RGBA / palette
        # images don't break the 3-channel Normalize.
        image = Image.open(file_path).convert("RGB")
        image = np.array(image)
        image = self.augmentation(image = image)["image"]


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
            persistent    : bool = True,
            generator     : Union[torch.Generator, None] = None
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
            persistent_workers = persistent_workers,
            worker_init_fn     = seed_worker if total_workers > 0 else None,
            generator          = generator
        )
        return data
    
if __name__ == "__main__":
    print("Basic Dataset Classification")

    dataset = DirectoryClassification(
        root_dir_path = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\arcturus_lychee\\.tests\\example_dataset\\LeafDataset\\validation",
        augmentation  = [
            A.HorizontalFlip(p = 0.5)
        ] 
    )

    dataset_loader = dataset.create_dataloader(1, 0, 'cpu')
    for image, label in dataset_loader:
        print(image.shape)
        print(label.shape)
        print(label)
        break
    
    
