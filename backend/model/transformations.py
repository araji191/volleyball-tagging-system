import torchvision.transforms.v2 as T
import torch

# Training
SIZE = (146, 32)  # closer to real ratio

train_transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),

    T.Resize(SIZE),  # acceptable for now

    T.RandomHorizontalFlip(p=0.25),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05)),

    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation/Test
val_transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize(SIZE),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])