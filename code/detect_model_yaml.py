import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import mediapipe as mp

# Define pose recognition model
class PoseNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super(PoseNet, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet.fc = nn.Linear(512, num_keypoints * 3)  # Predict (x, y, z) coordinates
    
    def forward(self, x):
        return self.resnet(x)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseNet(num_keypoints=17).to(device)
model.load_state_dict(torch.load("pose_model.pth", map_location=device))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.6  

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Convert landmark coordinates
            keypoints = []
            for lm in landmarks:
                keypoints.extend([lm.x, lm.y, lm.z])

            keypoints_tensor = torch.tensor(keypoints).view(1, -1).to(device)
            
            with torch.no_grad():
                outputs = model(keypoints_tensor)
            
            # Calculate confidence score
            confidence = torch.sigmoid(outputs).mean().item()

            if confidence >= CONFIDENCE_THRESHOLD:
                feedback = "Good posture!"
                color = (0, 255, 0)
            else:
                feedback = "Adjust your posture!"
                color = (0, 0, 255)

            cv2.putText(image, feedback, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Visualize landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Pose Feedback', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
