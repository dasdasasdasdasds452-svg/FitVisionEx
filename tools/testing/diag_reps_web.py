import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(r"C:\Users\tonkla\Downloads\VID_20260223_195246.mp4")

# Variables mirroring the web logic exactly
upThreshold = 140
downThreshold = 75
repState = "up"
localRepCount = 0

print("Simulating Web App Repetition Engine...")
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    # Optional: simulate 10-20 FPS by dropping frames to mirror browser performance
    # if frame_count % 2 != 0: continue
        
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Get coords
        l_sh = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_el = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wr = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        r_sh = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_el = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wr = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        feature_0 = calculate_angle(l_sh, l_el, l_wr)
        feature_1 = calculate_angle(r_sh, r_el, r_wr)
        
        mainAngle = (feature_0 + feature_1) / 2
        
        # Exact copy of the TS logic
        if mainAngle > upThreshold:
            if repState == "down":
                localRepCount += 1
                print(f"[{frame_count}] REP COUNTED: {localRepCount} | Angle reached: {mainAngle:.1f}")
            repState = "up"
        elif mainAngle < downThreshold:
            repState = "down"

print(f"\n--- FINAL RESULTS ---")
print(f"Total Reps Counted: {localRepCount}")

cap.release()
