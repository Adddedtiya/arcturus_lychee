import cv2            as cv
import albumentations as A

def light_aug() -> list:
    
    # create a list
    augmentations = [
        A.HorizontalFlip(p = 0.5),
        A.VerticalFlip(p = 0.5),
        A.Rotate(limit = 30, border_mode = cv.BORDER_REFLECT_101, p = 0.5),
    ]

    # return the list
    return augmentations

def medium_aug() -> list:

    # we build based on the light version
    augmentations = light_aug() + [
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

    return augmentations

def heavy_aug() -> list:

    # we build based on the medium version
    augmentations = medium_aug() + [
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

    return augmentations