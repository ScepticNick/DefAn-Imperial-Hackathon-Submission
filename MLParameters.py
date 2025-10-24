"""This is the code we used to train the AI to recognise rubbish. It utilises a preexisting convolutional neural netowrk that was further trained on a rubbish dataset. 
These parameters were downloaded and used in the main.py file.
If you want the parameters for yourself, simply use the following kaggle dataset; first download and then run our code. Thanks!
https://www.kaggle.com/datasets/mostafaabla/garbage-classification"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Dataset and loading 
path = r"C:\Users\leek37\Desktop\GarbageData\Garbage classification\Garbage classification"

transformations = transforms.Compose([
    transforms.Resize((128, 128)),   # smaller size = faster training
    transforms.ToTensor()
])

dataset = ImageFolder(path, transform=transformations)

# train test split
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=16, num_workers=0)

num_classes = len(dataset.classes)

# Model
class GarbageClassifier(nn.Module):
    def _init_(self, num_classes):
        super()._init_()
        self.network = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)


if _name_ == "_main_":
    # Force CPU only
    print(num_classes)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Model, optimizer
    model = GarbageClassifier(num_classes).to(device)
    print("Model Initialised")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print("optimiser initialised")
    num_epochs = 4
    print('epochs initialised')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        print("model training")
        train_loss = 0
        i = 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = F.cross_entropy(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'{i} Images done of this epoch')
            i+=1
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = F.cross_entropy(out, labels)
                acc = (torch.max(out, 1)[1] == labels).float().mean()

                val_loss += loss.item()
                val_acc += acc.item()

        # Average results
        train_loss /= len(train_dl)
        val_loss /= len(test_dl)
        val_acc /= len(test_dl)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")

    # Save model parameters
    save_dir = r"C:\Users\leek37\Desktop\Hackathon"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "garbage_classifier_v2.pth")

    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
