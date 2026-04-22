import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split

# ----------------------------
# CONFIG
# ----------------------------
DATA_ROOT = "keypoints_json3"
BATCH_SIZE = 128
EPOCHS = 500
LR = 1e-3
PATIENCE = 100
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CLASS_MAP = {
    "spike": 0,
    "defense": 1,
    "block": 2,
    "set": 3,
    "serve": 4
}

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def get_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def normalize_keypoints(kp):
    kp = np.array(kp)

    xy = kp[:, :2]
    conf = kp[:, 2:]

    # mask low confidence joints
    xy[conf.squeeze() < 0.3] = 0

    # center (hips)
    center = (xy[11] + xy[12]) / 2
    xy = xy - center

    # scale (torso)
    torso = np.linalg.norm((xy[5] + xy[6]) / 2)
    if torso > 0:
        xy = xy / torso

    # flip direction
    if xy[6][0] < xy[5][0]:
        xy[:, 0] *= -1

    # ------------------------
    # ANGLE FEATURES
    # ------------------------
    angles = []

    # elbows
    angles.append(get_angle(xy[5], xy[7], xy[9]))   # left arm
    angles.append(get_angle(xy[6], xy[8], xy[10]))  # right arm

    # knees
    angles.append(get_angle(xy[11], xy[13], xy[15]))
    angles.append(get_angle(xy[12], xy[14], xy[16]))

    # shoulders
    angles.append(get_angle(xy[7], xy[5], xy[11]))
    angles.append(get_angle(xy[8], xy[6], xy[12]))

    angles = np.array(angles)

    return np.concatenate([xy.flatten(), conf.flatten(), angles])


# ----------------------------
# DATASET
# ----------------------------
class PoseDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with open(path) as f:
            data = json.load(f)

        kp = normalize_keypoints(data["keypoints"])

        return torch.tensor(kp, dtype=torch.float32), torch.tensor(label)


# ----------------------------
# DATA LOADING
# ----------------------------
def load_all_samples(splits=["train", "valid"]):
    all_samples = []

    for split in splits:
        split_dir = os.path.join(DATA_ROOT, split)
        for class_name in CLASS_MAP:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for file in os.listdir(class_dir):
                if file.endswith(".json"):
                    all_samples.append((
                        os.path.join(class_dir, file),
                        CLASS_MAP[class_name]
                    ))

    return all_samples


def print_class_dist(name, samples):
    inv_name = {v: k for k, v in CLASS_MAP.items()}
    counts = Counter(s[1] for s in samples)
    print(f"\n{name} ({len(samples)} samples):")
    for i in range(len(CLASS_MAP)):
        print(f"  {inv_name[i]}: {counts.get(i, 0)}")


# ----------------------------
# CLASS WEIGHTS
# ----------------------------
def compute_class_weights(samples):
    counts = Counter(s[1] for s in samples)
    total = sum(counts.values())
    num_classes = len(CLASS_MAP)
    weights = torch.tensor(
        [total / (counts.get(i, 1) * num_classes) for i in range(num_classes)],
        dtype=torch.float32
    )
    return weights


# ----------------------------
# MODEL
# ----------------------------
class PoseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(57, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# TRAIN / EVAL
# ----------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


# ----------------------------
# MAIN
# ----------------------------
def main():
    # load train + valid splits together and re-split with stratification
    all_samples = load_all_samples(["train", "valid"])
    labels = [s[1] for s in all_samples]

    train_samples, val_samples = train_test_split(
        all_samples,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    print_class_dist("Train", train_samples)
    print_class_dist("Val", val_samples)

    train_dataset = PoseDataset(train_samples)
    valid_dataset = PoseDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    # weighted loss
    class_weights = compute_class_weights(train_samples).to(DEVICE)
    print(f"\nClass weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = PoseClassifier().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nTraining on {DEVICE}...\n")

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_acc = evaluate(model, valid_loader)
        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}  |  LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "pose_classifier_best.pt")
            print(f"  ✅ New best — saved pose_classifier_best.pt")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{PATIENCE})")

        print("-" * 40)

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}. Best val acc: {best_val_acc:.4f}")
            break

    torch.save(model.state_dict(), "pose_classifier_final.pt")
    print(f"\nDone. Best val acc: {best_val_acc:.4f}")
    print("Checkpoints: pose_classifier_best.pt | pose_classifier_final.pt")


if __name__ == "__main__":
    main()