from torchvision import transforms
from typing import Tuple

def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
   
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
