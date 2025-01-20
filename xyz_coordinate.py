import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("A")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            h, w, _ = frame.shape 
            cx, cy, cz = int(landmark.x * w), int(landmark.y * h), landmark.z

            cv2.putText(frame, f'({cx}, {cy}, {cz:.2f})', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
            print(f"Landmark {idx}: x={cx}, y={cy}, z={cz:.2f}")

    cv2.imshow('Pose Estimation', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
