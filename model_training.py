from pathlib import Path

from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import DataLoader

from transformations import train_transform, val_transform
from cnn import CNN

path = Path("./VolleyballDataset")

if not path.exists():
   print("Dataset not found")
else:
    print("Volleyball Dataset Found")

BATCH = 32

torch.manual_seed(12345)

train_set = ImageFolder(path / "train", train_transform)
val_set = ImageFolder(path / "valid", val_transform)

# Data Loaders
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH)

# TRAINING LOOP

# EVALUATE (on the validation set):
# accuracy

# SAVE MODEL
#torch.save(model.state_dict(), path_to_cards / "model.pt")
