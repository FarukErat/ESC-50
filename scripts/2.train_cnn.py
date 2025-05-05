import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ESC50Dataset(Dataset):
    def __init__(self, meta_csv, spectrogram_dir, folds, transform=None, selected_classes=None):
        self.meta = pd.read_csv(meta_csv)
        self.meta = self.meta[self.meta.fold.isin(folds)]

        if selected_classes is not None:
            self.meta = self.meta[self.meta.target.isin(selected_classes)].reset_index(drop=True)
            target_mapping = {orig: new for new, orig in enumerate(sorted(selected_classes))}
            self.meta['target'] = self.meta['target'].map(target_mapping)

        self.spect_dir = spectrogram_dir
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        fname = row.filename.replace('.wav', '_spectrogram.png')
        img_path = os.path.join(self.spect_dir, fname)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = int(row.target)
        return image, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Updated input size for the first Linear layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 43, 256),  # Adjusted input size
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total


def per_class_accuracy(model, loader, device, num_classes):
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1
    acc_dict = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in range(num_classes)}
    return acc_dict


def load_label_map(meta_csv, selected_classes):
    df = pd.read_csv(meta_csv)
    df = df[df.target.isin(selected_classes)]
    df = df[['target', 'category']].drop_duplicates()
    target_to_new_idx = {orig: new for new, orig in enumerate(sorted(selected_classes))}
    mapping = {target_to_new_idx[row.target]: row.category for _, row in df.iterrows()}
    return mapping


def main():
    batch_size = 32

    meta_csv = os.path.join('meta', 'esc50.csv')
    spect_dir = os.path.join('mel_spectrograms')

    selected_class_names = [
        "can_opening", "siren", "crying_baby", "frog", "sea_waves",
        "crickets", "thunderstorm", "brushing_teeth", "door_wood_knock", "clock_alarm"
    ]

    df = pd.read_csv(meta_csv)
    df = df[df['category'].isin(selected_class_names)]
    selected_targets = sorted(df['target'].unique())
    target_to_new_idx = {old: new for new, old in enumerate(selected_targets)}
    category_map = {target_to_new_idx[row.target]: row.category for _, row in df.iterrows() if row.target in selected_targets}

    fold = 1
    epochs = 20

    train_folds = [f for f in range(1, 6) if f != fold]
    val_fold = fold

    transform = transforms.Compose([
        transforms.Resize((64*2, 173*2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = ESC50Dataset(meta_csv, spect_dir, train_folds, transform, selected_classes=selected_targets)
    val_dataset = ESC50Dataset(meta_csv, spect_dir, [val_fold], transform, selected_classes=selected_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode='triangular'
)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f} - "
              f"Learning Rate: {current_lr:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(f"best_model_fold{fold}.pth"))

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")

    print("Evaluating best model on test fold...")
    model.load_state_dict(torch.load(os.path.join(f"best_model_fold{fold}.pth")))
    test_loss, test_acc = evaluate(model, val_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}\n")

    print("Per-class accuracy (sorted descending):")
    acc_dict = per_class_accuracy(model, val_loader, device, num_classes=10)
    for idx, acc in sorted(acc_dict.items(), key=lambda x: x[1], reverse=True):
        category = category_map.get(idx, f"Class {idx}")
        print(f"  {idx:02d} - {category:20s}: {acc:.4f}")


if __name__ == '__main__':
    main()
