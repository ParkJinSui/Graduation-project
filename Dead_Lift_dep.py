import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

# Mediapipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize repetition counter
counter = 0
stage = None  # 'down' or 'up'
correct = False  # Indicates if the posture is correct during the lift

# Function to calculate the angle
def calculate_angle(a, b, c):
    a = np.array(a[:2])  # Hip
    b = np.array(b[:2])  # Knee
    c = np.array(c[:2])  # Ankle

    radians = np.arccos(
        np.dot(a - b, c - b) / (np.linalg.norm(a - b) * np.linalg.norm(c - b))
    )
    angle = np.degrees(radians)
    return angle

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Convert image to numpy array
        image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        try:
            landmarks = results.pose_landmarks.landmark

            # Function to get 3D coordinates
            def get_3d_coords(landmark):
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                depth = int(depth_frame.get_distance(x, y) * 1000)  # Convert to mm
                return [x, y, f'{depth} mm']

            # Draw all skeleton points with XYZ coordinates
            for idx, landmark in enumerate(landmarks):
                coords = get_3d_coords(landmark)
                pos = (coords[0], coords[1])
                cv2.putText(image, f'{idx}: {coords}', pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

            # Left joint coordinates
            shoulder = get_3d_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            hip = get_3d_coords(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
            knee = get_3d_coords(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
            ankle = get_3d_coords(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

            # Calculate angles
            hip_angle = calculate_angle(shoulder, hip, knee)
            knee_angle = calculate_angle(hip, knee, ankle)

            # Determine repetition count and correctness
            if hip_angle > 160 and knee_angle > 160:
                stage = "up"
                correct = False

            if hip_angle < 100 and knee_angle < 120 and stage == "up":
                stage = "down"
                correct = True
                counter += 1

            # Draw angles on image
            cv2.putText(image, f'Hip: {int(hip_angle)}', (hip[0], hip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, f'Knee: {int(knee_angle)}', (knee[0], knee[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        except:
            pass

        # Draw Mediapipe landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display repetition count and stage
        cv2.putText(image, f'Reps: {counter}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f'Stage: {stage}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f'Correct: {"Yes" if correct else "No"}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Deadlift Tracker with RealSense', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
