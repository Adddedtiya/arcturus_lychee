# Datasets and augmentations. These are examples of project code.
# For a new project, add a dataset file here, or change these files.

# A classification dataset with one directory for each class
from arcturus_lychee.datasets.basic_classification_dataset import (
    DirectoryClassification,
)

# Three levels of image augmentations
from arcturus_lychee.datasets.generic_augmentations import (
    light_aug,
    medium_aug,
    heavy_aug,
)
