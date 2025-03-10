import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pypose as pp

# Define paths
dataset_path = "dataset"
image_path = os.path.join(dataset_path, "images")
label_json = os.path.join(dataset_path, "labels.json")

# Load labels
with open(label_json, "r") as f:
    labels = json.load(f)

# Custom dataset class
class PoseDataset(Dataset):
    def __init__(self, image_path, labels, transform=None):
        self.image_path = image_path
        self.labels = list(labels.items())
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_file, data = self.labels[idx]
        img_path = os.path.join(self.image_path, img_file)
        image = Image.open(img_path).convert('RGB')
        keypoints = [list(lm.values()) for lm in data["landmarks"].values()]
        keypoints = torch.tensor(keypoints).view(-1)  # Flatten keypoints
        
        if self.transform:
            image = self.transform(image)
        
        return image, keypoints

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create dataset and dataloader
dataset = PoseDataset(image_path, labels, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define neural network model
class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet.fc = nn.Linear(512, len(labels[list(labels.keys())[0]]["landmarks"]) * 3)  # Predict (x, y, z)
    
    def forward(self, x):
        return self.resnet(x)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, keypoints in dataloader:
        images, keypoints = images.to(device), keypoints.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Save trained model
torch.save(model.state_dict(), "pose_model.pth")
print("Model training complete and saved.")
