import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Dataset
# ----------------------------
class SpectrogramDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        filename = self.annotations.iloc[idx, 0]
        label = self.annotations.iloc[idx, 2]  # 'target'
        img_name = os.path.splitext(filename)[0] + "_spectrogram.png"
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# ----------------------------
# Model
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 16, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 32, 56, 56]
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------------------
# Main Training Function
# ----------------------------
def main():
    CSV_PATH = 'meta/esc50.csv'
    IMG_DIR = 'spectrograms'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = SpectrogramDataset(CSV_PATH, IMG_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleCNN(num_classes=50)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    main()
