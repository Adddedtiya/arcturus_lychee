"""Three levels of image augmentations for albumentations.

Each level contains the augmentations of the level before it:
light_aug() < medium_aug() < heavy_aug(). Each function returns a new list.
DirectoryClassification puts the list between the resize and the crop.
"""

import cv2            as cv
import albumentations as A


def light_aug() -> list[A.BasicTransform]:
    """Return flips and a small rotation."""
    return [
        A.HorizontalFlip(p = 0.5),
        A.VerticalFlip(p = 0.5),
        A.Rotate(limit = 30, border_mode = cv.BORDER_REFLECT_101, p = 0.5),
    ]


def medium_aug() -> list[A.BasicTransform]:
    """Return light_aug(), and also affine changes and color changes."""
    return light_aug() + [
        A.Affine(
            scale             = (0.9, 1.1),
            translate_percent = (0.0, 0.05),
            shear             = (-8, 8),
            border_mode       = cv.BORDER_REFLECT_101,
            p                 = 0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit = 0.2,
            contrast_limit   = 0.2,
            p                = 0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit = 10,
            sat_shift_limit = 20,
            val_shift_limit = 10,
            p               = 0.3,
        ),
    ]


def heavy_aug() -> list[A.BasicTransform]:
    """Return medium_aug(), and also dropout, noise or blur, and gamma changes."""
    return medium_aug() + [
        A.CoarseDropout(
            num_holes_range   = (1, 4),
            hole_height_range = (0.05, 0.15),
            hole_width_range  = (0.05, 0.15),
            fill              = 0,
            p                 = 0.3,
        ),
        A.OneOf([
            A.GaussNoise(std_range = (0.1, 0.2)),
            A.GaussianBlur(blur_limit = (3, 7)),
            A.MotionBlur(blur_limit = (3, 7)),
        ], p = 0.3),
        A.RandomGamma(gamma_limit = (80, 120), p = 0.3),
    ]
