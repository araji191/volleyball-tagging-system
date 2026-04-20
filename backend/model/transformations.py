import torchvision.transforms.v2 as T
import torch
import torchvision.transforms.functional as F

import torchvision.transforms.v2 as T
import torch

SIZE = (146, 38)

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

#TARGET_HEIGHT = 288
#TARGET_WIDTH  = 64

"""
class ResizePad:
    def __init__(self, target_h, target_w):
        self.target_h = target_h
        self.target_w = target_w

    def __call__(self, img):
        _, h, w = img.shape
        scale = min(self.target_h / h, self.target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = F.resize(img, [new_h, new_w])
        pad_h = self.target_h - new_h
        pad_w = self.target_w - new_w
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left
        img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom])
        return img

resize_pad = ResizePad(TARGET_HEIGHT, TARGET_WIDTH)

# Training
train_transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    resize_pad,
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation / Test
val_transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    resize_pad,
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])"""

