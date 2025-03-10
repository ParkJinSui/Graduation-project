import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet.fc = nn.Linear(512, 3 * len(landmarks))
    
    def forward(self, x):
        return self.resnet(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseNet().to(device)
model.load_state_dict(torch.load("pose_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
    
    predicted_keypoints = outputs.view(-1, 3).cpu().numpy()
    for i, keypoint in enumerate(predicted_keypoints):
        cv2.putText(frame, f'Landmark {i+1}: ({keypoint[0]:.2f}, {keypoint[1]:.2f})', 
                    (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
