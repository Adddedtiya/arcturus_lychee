import os
import torch
import numpy as np

from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset
from typing           import Optional

from arcturus_lychee.helpers import scan_directory_for_images


class DirectoryClassification(Dataset):
    """An image classification dataset with one directory for each class.

    The layout of root_dir_path:

        root_dir_path/
            class_a/   image files
            class_b/   image files

    The class index is the position of the class name in the sorted list of
    directory names. self.class_names keeps this order.

    Each image goes through this pipeline:

      1. Resize to 256 x 256.
      2. The augmentations in the augmentation list, if the list is not None.
      3. A crop to 224 x 224. If training is True, the crop position is random.
         If training is False, the crop is at the center. Thus the evaluation
         gives the same result each time.
      4. Normalize with the ImageNet mean and standard deviation.
      5. Conversion to a CHW float tensor.

    Give the augmentation list only for the training set.
    """

    def __init__(
            self,
            root_dir_path : str,
            augmentation  : Optional[list] = None,
            training      : bool           = True,
            seed          : Optional[int]  = None,
        ) -> None:
        super().__init__()

        self.root_dir = root_dir_path
        self.training = training

        # The dataset: a list of (file path, class index)
        self.dataset_list : list[tuple[str, int]] = []

        self.class_names = self._scan_for_folders()
        self.total_class = len(self.class_names)

        for class_index, class_name in enumerate(self.class_names):
            class_files = scan_directory_for_images(
                root_dir = os.path.join(self.root_dir, class_name)
            )
            for file_path in class_files:
                self.dataset_list.append((file_path, class_index))

        # The albumentations pipeline: a numpy HWC uint8 image in, a CHW float tensor out.
        # Normalize must come before ToTensorV2. ToTensorV2 only changes HWC to CHW, and does not scale.
        crop = A.RandomCrop(224, 224) if training else A.CenterCrop(224, 224)

        augmentation_stack  = [A.Resize(256, 256)]
        augmentation_stack += augmentation if augmentation else []
        augmentation_stack += [
            crop,
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std  = [0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]

        self.augmentation = A.Compose(augmentation_stack, seed = seed)

    def _scan_for_folders(self) -> list[str]:
        """Return the sorted names of the class directories."""
        folder_names = [
            name for name in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, name))
        ]
        folder_names.sort()
        return folder_names

    def __len__(self) -> int:
        return len(self.dataset_list)

    def __getitem__(self, index : int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the image [C, H, W] and the class index (a tensor with no dimensions)."""
        file_path, class_index = self.dataset_list[index]

        # convert("RGB") gives 3 channels also for grayscale, RGBA, and palette images.
        # The Normalize step needs 3 channels.
        image = Image.open(file_path).convert("RGB")
        image = np.array(image)
        image = self.augmentation(image = image)["image"]

        label = torch.tensor(class_index)
        return image, label


if __name__ == "__main__":
    # Demo: load one batch. Change the path to a directory with one subdirectory for each class.
    from arcturus_lychee.helpers import build_dataloader

    DATASET_PATH = "dataset_path/train"

    dataset = DirectoryClassification(
        root_dir_path = DATASET_PATH,
        augmentation  = [A.HorizontalFlip(p = 0.5)],
        training      = True,
    )
    print(f"Classes: {dataset.class_names}")
    print(f"Images : {len(dataset)}")

    loader = build_dataloader(dataset, batch_size = 1)
    for image, label in loader:
        print(f"Image shape: {tuple(image.shape)}")
        print(f"Label      : {label.tolist()}")
        break
