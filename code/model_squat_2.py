import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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
        self.labels = list(labels.items())  # 레이블 리스트
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_file, data = self.labels[idx]
        img_path = os.path.join(self.image_path, img_file)
        image = Image.open(img_path).convert('RGB')
        
        # 각 이미지에 대한 동작의 레이블 (스쿼트 단계)
        stage = data['stage']  # 준비, 스쿼트 중, 앉은 자세, 일어나는 자세 등
        
        if self.transform:
            image = self.transform(image)
        
        return image, stage  # 이미지와 해당하는 단계 레이블을 반환

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
    def __init__(self, num_classes=4):  # 4가지 동작 단계
        super(PoseNet, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)  # 출력층을 클래스 수에 맞게 수정
    
    def forward(self, x):
        return self.resnet(x)  # 분류 예측

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseNet(num_classes=4).to(device)  # 4가지 스쿼트 단계 예측
criterion = nn.CrossEntropyLoss()  # 분류 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, stages in dataloader:
        images, stages = images.to(device), stages.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        # CrossEntropyLoss는 정수형 레이블을 기대하므로, stages는 레이블 인덱스여야 함
        loss = criterion(outputs, stages)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Save trained model
torch.save(model.state_dict(), "pose_model.pth")
print("Model training complete and saved.")
