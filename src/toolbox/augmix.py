import albumentations as A
from PIL import Image
import numpy as np

def augmix(image, severity=3, width=3, depth=1, alpha=1.0, p=0.5):
    # Define the base augmentation pipeline
    transform_base = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.RandomGamma(gamma_limit=(80, 120)),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5),
        ], p=0.7),
        A.OneOf([
            A.Blur(),
            A.GaussianBlur(),
            A.MedianBlur(),
            A.GaussNoise(),
        ], p=0.7),
    ], p=p)

    # Define the augmentations for each operation
    transforms_aug = [
        A.RandomBrightnessContrast(brightness_limit=0.1*severity, contrast_limit=0.1*severity, always_apply=True),
        A.RandomGamma(),
        A.HueSaturationValue(p=1.0, hue_shift_limit=3*severity, sat_shift_limit=3*severity, val_shift_limit=3*severity),
        A.Blur(blur_limit=3+int(severity/3), p=1.0),
        A.GaussianBlur(p=1.0),
        A.MedianBlur(p=1.0),
        A.GaussNoise(var_limit=(10.0**(-severity/2), 10.0**(severity/2)), p=1.0),
    ]
    
    # Create the augmented image
    for i in range(depth):
        transform_aug = A.Compose([
            transforms_aug[i % len(transforms_aug)],
            transform_base,
            transforms_aug[(i+1) % len(transforms_aug)],
        ], p=p)
        image = transform_aug(image=image)['image']
    
    # Mix the original image with the augmented image
    transform_mix = A.Compose([
        A.Blur(p=1.0),
        A.GaussianBlur(p=1.0),
        A.MedianBlur(p=1.0),
        A.MotionBlur(p=1.0),
        A.CoarseDropout(p=1.0),
    ], p=p)
    
    image_aug = transform_mix(image=image)['image']
    image= Image.fromarray(image)
    image_aug= Image.fromarray(image_aug)
    image_mix = Image.blend(image, image_aug, alpha)

    return np.array(image_mix)
