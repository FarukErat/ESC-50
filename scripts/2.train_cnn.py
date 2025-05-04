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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a CNN on the ESC-50 dataset using spectrogram images"
    )
    parser.add_argument(
        "--data-dir", type=str, default=".",
        help="Root directory of ESC-50 (contains 'meta' and 'spectrograms')"
    )
    parser.add_argument(
        "--fold", type=int, default=1,
        help="Which fold to use as test (1-5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )
    return parser.parse_args()


class ESC50Dataset(Dataset):
    def __init__(self, meta_csv, spectrogram_dir, folds, transform=None):
        self.meta = pd.read_csv(meta_csv)
        self.meta = self.meta[self.meta.fold.isin(folds)].reset_index(drop=True)
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
    def __init__(self, num_classes=50):
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
        # Note: input size 64x173 -> after 3 poolings: 8x21
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 21, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
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


def load_label_map(meta_csv):
    df = pd.read_csv(meta_csv)
    label_map = df[['target', 'category']].drop_duplicates().set_index('target')['category'].to_dict()
    return label_map


def main():
    args = parse_args()

    meta_csv = os.path.join(args.data_dir, 'meta', 'esc50.csv')
    spect_dir = os.path.join(args.data_dir, 'spectrograms')

    train_folds = [f for f in range(1, 6) if f != args.fold]
    val_fold = args.fold

    transform = transforms.Compose([
        transforms.Resize((64, 173)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = ESC50Dataset(meta_csv, spect_dir, train_folds, transform)
    val_dataset = ESC50Dataset(meta_csv, spect_dir, [val_fold], transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device(args.device)
    model = SimpleCNN(num_classes=50).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.data_dir, f"best_model_fold{args.fold}.pth"))

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")

    print("Evaluating best model on test fold...")
    model.load_state_dict(torch.load(os.path.join(args.data_dir, f"best_model_fold{args.fold}.pth")))
    test_loss, test_acc = evaluate(model, val_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}\n")

    # Per-category accuracy sorted descending
    print("Per-class accuracy (sorted descending):")
    acc_dict = per_class_accuracy(model, val_loader, device, num_classes=50)
    label_map = load_label_map(meta_csv)
    # Sort by accuracy value descending
    for idx, acc in sorted(acc_dict.items(), key=lambda x: x[1], reverse=True):
        category = label_map.get(idx, f"Class {idx}")
        print(f"  {idx:02d} - {category:20s}: {acc:.4f}")

if __name__ == '__main__':
    main()
