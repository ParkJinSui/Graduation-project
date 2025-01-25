import cv2
import mediapipe as mp
import numpy as np

# Mediapipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize repetition counter
counter = 0
stage = None  # 'down' or 'up'

# Function to calculate the angle
def calculate_angle(a, b, c):
    a = np.array(a)  # Shoulder
    b = np.array(b)  # Elbow
    c = np.array(c)  # Wrist

    radians = np.arccos(
        np.dot(a - b, c - b) / (np.linalg.norm(a - b) * np.linalg.norm(c - b))
    )
    angle = np.degrees(radians)
    return angle

# Capture webcam input
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Flip image and process with Mediapipe
    image = cv2.flip(frame, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        # Left joint coordinates
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Calculate arm angle
        angle = calculate_angle(shoulder, elbow, wrist)

        # Determine repetition count
        if angle > 160:
            stage = "up"
        if angle < 60 and stage == "up":
            stage = "down"
            counter += 1

        # Draw the angle on the image
        elbow_coords = (int(elbow[0] * image.shape[1]), int(elbow[1] * image.shape[0]))
        cv2.putText(image, f'{int(angle)}°', 
                    elbow_coords, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    except:
        pass

    # Draw Mediapipe landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display repetition count and stage
    cv2.putText(image, f'Reps: {counter}', 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f'Stage: {stage}', 
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Bench Press Tracker', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
