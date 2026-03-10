from torchvision import transforms
from typing import Tuple

def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns training and validation transforms.
    
    Args:
        img_size (int): Target image size.
        
    Returns:
        train_transforms, val_transforms
    """
    # CLIP mean and std
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transforms, val_transforms
"""
   src/data/transforms.py — Preparing Images
The theory: why transform images at all?
Neural networks expect images as numbers, not raw pixels. More importantly, a model that has only seen a jacket photographed from exactly one angle, in exactly one lighting, will fail when it sees the same jacket photographed differently. Data augmentation artificially creates variety so the model learns robust features.

What the code does

Training:
  RandomResizedCrop  →  randomly zoom into a region of the image (90–100% of it)
  RandomHorizontalFlip  →  randomly mirror the image left–right
  ColorJitter  →  slightly shift brightness, contrast, colour
  ToTensor  →  convert from a photo (0–255) to numbers (0.0–1.0)
  Normalize  →  shift numbers so their average and spread matches CLIP's expectations
The Normalize step uses very specific numbers — (0.48145466, 0.4578275, 0.40821073) for mean and similar for std. These aren't arbitrary. They were calculated from the 400 million images CLIP was originally trained on. Reusing them is important because CLIP learned to expect inputs in that exact range.

For validation, there's no random cropping or flipping — you always want the same deterministic crop so results are reproducible and fair.
"""