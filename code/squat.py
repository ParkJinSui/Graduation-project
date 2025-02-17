import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    
    ab = [a[0] - b[0], a[1] - b[1]]
    cb = [c[0] - b[0], c[1] - b[1]]
    
    dot_product = ab[0] * cb[0] + ab[1] * cb[1]
    magnitude_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    magnitude_cb = math.sqrt(cb[0]**2 + cb[1]**2)
    
    angle = math.acos(dot_product / (magnitude_ab * magnitude_cb))
    return math.degrees(angle)

cap = cv2.VideoCapture(0)
squat_count = 0
stage = None  # None, 'down', 'up'
ready = False  # Indicates if the user is in the ready position
correct = False

def is_ready_position(hip_angle, knee_angle):
    return (140 <= hip_angle <= 180 and 140 <= knee_angle <= 180)

def is_correct_squat(hip_angle, knee_angle):
    return (70 <= knee_angle <= 190 and hip_angle < 190)

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
        
        feedback = ""
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

            hip_angle = calculate_angle(shoulder, hip, knee)
            knee_angle = calculate_angle(hip, knee, ankle)

            # Display angles
            cv2.putText(image, f'Hip: {int(hip_angle)}',
                        tuple(np.multiply([hip.x, hip.y], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.putText(image, f'Knee: {int(knee_angle)}',
                        tuple(np.multiply([knee.x, knee.y], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Check for ready position
            if is_ready_position(hip_angle, knee_angle):
                ready = True
                if stage == 'up':
                    stage = None  # Reset stage after completing a squat
                cv2.putText(image, "Ready Position", (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Squat logic
            if ready:
                if hip_angle < 90 and knee_angle < 90:
                    if stage != 'down':
                        stage = 'down'
                        correct = True
                        
                if stage == 'down' :
                    if 80 < hip_angle < 90 and 70 < knee_angle < 90 :
                        feedback = "Good posture!"
                        correct = True
                    elif 90 < hip_angle or 90 < knee_angle :
                        feedback = "Incorrect posture! Down your hip"
                    elif 80 > hip_angle or 70 > knee_angle :
                        feedback = "Incorrect posture! Up your hip"
                    
                    
                if hip_angle > 160:
                    if stage == 'down' and correct :
                        if is_correct_squat(hip_angle, knee_angle):  # Count only if posture is correct
                            stage = 'up'
                            squat_count += 1
                        elif stage == 'down' and not correct :
                            feedback = "Incorrect posture."

            # Display squat count
            cv2.putText(image, f'Squats: {squat_count}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
             # Display stage (up or down)
            cv2.putText(image, f'Stage: {stage}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display feedback
        if feedback:
            cv2.putText(image, feedback, (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        
                        

        cv2.imshow('Mediapipe Squat Analyzer', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
