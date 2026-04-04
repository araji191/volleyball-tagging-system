from pathlib import Path

from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import DataLoader

from model.transformations import train_transform, val_transform
from model.cnn import CNN

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(device)

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
#torch.save(model.state_dict(), "weights/action_model.pt")
