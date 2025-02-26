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
deadlift_count = 0
stage = None  # 'down', 'up'
correct = False  # Indicates if the posture is correct during the lift

# Define thresholds for posture
def is_ready_position(hip_angle, knee_angle):
    return 160 <= hip_angle <= 180 and 160 <= knee_angle <= 180

def is_correct_deadlift(hip_angle, knee_angle, back_angle):
    return hip_angle <= 120 and 150 <= back_angle <= 180 and 80 <= knee_angle <= 120

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
            ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]

            hip_angle = calculate_angle(shoulder, hip, knee)
            knee_angle = calculate_angle(hip, knee, ankle)
            back_angle = calculate_angle(ear, shoulder, hip)

            if back_angle < 140:
                feedback = "Straighten your back!"

            # Display angles
            cv2.putText(image, f'Hip: {int(hip_angle)}',
                        tuple(np.multiply([hip.x, hip.y], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.putText(image, f'Knee: {int(knee_angle)}',
                        tuple(np.multiply([knee.x, knee.y], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.putText(image, f'Back: {int(back_angle)}',
                        tuple(np.multiply([shoulder.x, shoulder.y], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Check ready position
            if is_ready_position(hip_angle, knee_angle):
                stage = None  # Reset stage after completing a lift
                feedback = "Ready Position - Start Deadlift"
                correct = False  # Reset correctness for new lift attempt

            # Deadlift logic
            if stage != 'down' and hip_angle < 100:  # Down phase
                stage = 'down'
                feedback = "Lowering - Check your form"
                correct = is_correct_deadlift(hip_angle, knee_angle, back_angle)

            elif stage == 'down' and hip_angle > 100:  # Up phase
                stage = 'up'
                
                deadlift_count += 1
                
                if correct:  # Increment count only if lift was correct
                    feedback = "Good Deadlift!"
                else:
                    feedback = "Incorrect posture during lift"

            # Display deadlift count
            cv2.putText(image, f'Deadlifts: {deadlift_count}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display stage
            cv2.putText(image, f'Stage: {stage}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display feedback
        if feedback:
            cv2.putText(image, feedback, (10, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Mediapipe Deadlift Analyzer', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
