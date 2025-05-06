import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

Num_Classes = 50
fold = 1
epochs = 40
batch_size = 32

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
        if idx >= len(self.meta):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.meta)}")
        
        row = self.meta.iloc[idx]
        fname = row.filename.replace('.wav', '_spectrogram.png')
        img_path = os.path.join(self.spect_dir, fname)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")
        
        image = Image.open(img_path).convert('LA') 
        if self.transform:
            image = self.transform(image)
        label = int(row.target)
        return image, label

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        out = nn.GELU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(residual)
        out = nn.GELU()(out)
        return out

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(ImprovedCNN, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=1),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        self.layer4 = nn.Sequential(
            ResidualBlock(128, 256, stride=1),
            ResidualBlock(256, 256),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # First block - Initial feature extraction
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # Second block - Increase channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # Third block - Deep feature extraction
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Fourth block - High-level features
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),

            # Fifth block - Final feature refinement
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )
        # Calculate the input size for the linear layer
        self._to_linear = None
        self._find_linear_input(torch.zeros(1, 2, 128, 346))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, num_classes)
        )

    def _find_linear_input(self, x):
        self.features_output = self.features(x)
        self._to_linear = self.features_output.shape[1] * self.features_output.shape[2] * self.features_output.shape[3]

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
        outputs = model(imgs.to(device))
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

    meta_csv = os.path.join('meta', 'esc50.csv')
    spect_dir = os.path.join('mel_spectrograms')

    selected_class_names = [ 
        'dog', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm', 'door_wood_knock',
        'can_opening', 'crow', 'clapping', 'fireworks', 'chainsaw', 'airplane',
        'mouse_click', 'pouring_water', 'train', 'sheep', 'water_drops',
        'church_bells', 'clock_alarm', 'keyboard_typing', 'wind', 'footsteps', 'frog',
        'cow', 'brushing_teeth', 'car_horn', 'crackling_fire', 'helicopter',
        'drinking_sipping', 'rain', 'insects', 'laughing', 'hen', 'engine', 'breathing',
        'crying_baby', 'hand_saw', 'coughing', 'glass_breaking', 'snoring',
        'toilet_flush', 'pig', 'washing_machine', 'clock_tick', 'sneezing', 'rooster',
        'sea_waves', 'siren', 'cat', 'door_wood_creaks', 'crickets'
    ]

    df = pd.read_csv(meta_csv)
    # df = df[df['category'].isin(selected_class_names)] 
    selected_targets = sorted(df['target'].unique()) # Use targets present in the dataframe
    target_to_new_idx = {old: new for new, old in enumerate(selected_targets)}
    category_map = {target_to_new_idx[row.target]: row.category for _, row in df.iterrows() if row.target in selected_targets}
    global Num_Classes 
    Num_Classes = len(selected_targets) 
    print(f"Using {Num_Classes} classes.")

    train_folds = [f for f in range(1, 6) if f != fold]
    val_fold = fold
    train_transform = transforms.Compose([
        transforms.Resize((128, 346)),
        # Example: Add some image-based augmentation (use cautiously on spectrograms)
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0)), # Example: Random horizontal shift
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.1, 1.0), value=0), # <--- ADD Augmentation HERE
        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]) 
    ])
    val_transform = transforms.Compose([
        transforms.Resize((128, 346)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]) 
    ])

    train_dataset = ESC50Dataset(meta_csv, spect_dir, train_folds, train_transform, selected_classes=selected_targets)
    val_dataset = ESC50Dataset(meta_csv, spect_dir, [val_fold], val_transform, selected_classes=selected_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=min(4, os.cpu_count())) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ImprovedCNN(num_classes=Num_Classes).to(device)
    #model = SimpleCNN(num_classes=Num_Classes).to(device) 
    summary(model, (2, 128, 346))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2, 
        eta_min=1e-6
    )

    best_acc = 0.0
    output_dir = "." # Define where to save the model
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
    model_save_path = os.path.join(output_dir, f"best_model_fold{fold}.pth")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{epochs} - "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f} - "
              f"Learning Rate: {current_lr:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")

    if os.path.exists(model_save_path):
        print(f"\nEvaluating best model ({model_save_path}) on validation fold {fold}...")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        test_loss, test_acc = evaluate(model, val_loader, criterion, device)
        print(f"Validation loss: {test_loss:.4f}, Validation accuracy: {test_acc:.4f}\n")
        print("Per-class accuracy (sorted descending):")
        acc_dict = per_class_accuracy(model, val_loader, device, num_classes=Num_Classes) 
        for idx, acc in sorted(acc_dict.items(), key=lambda x: x[1], reverse=True):
            category = category_map.get(idx, f"Class {idx}") 
            print(f"  {idx:02d} - {category:20s}: {acc:.4f}")
    else:
        print(f"Model file not found at {model_save_path}, skipping final evaluation.")

if __name__ == '__main__':
    main()