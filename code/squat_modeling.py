import cv2
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseNet(num_classes=4).to(device)
model.load_state_dict(torch.load("pose_model.pth"))
model.eval()  


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


cap = cv2.VideoCapture(0)
squat_count = 0
stage = None  # None, 'down', 'up'
ready = False 
correct = False

def predict_stage(image):
    
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item() 

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


            predicted_stage = predict_stage(frame)
            if predicted_stage == 0:
                cv2.putText(image, "Ready Position", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif predicted_stage == 1:
                cv2.putText(image, "Squatting", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif predicted_stage == 2:
                cv2.putText(image, "Sitting Down", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif predicted_stage == 3:
                cv2.putText(image, "Standing Up", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Squat Pose Recognition', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
