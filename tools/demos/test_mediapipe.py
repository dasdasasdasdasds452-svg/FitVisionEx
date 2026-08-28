"""
Test MediaPipe Pose Estimation
"""
import cv2
import mediapipe as mp

def test_mediapipe():
    print("Testing MediaPipe Pose Estimation...")
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return False
    
    print("✅ Webcam opened successfully")
    print("Press 'q' to quit")
    
    frame_count = 0
    detected_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            detected_count += 1
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            cv2.putText(frame, f"Pose Detected! ({len(results.pose_landmarks.landmark)} landmarks)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No pose detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, f"Frame: {frame_count} | Detected: {detected_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('MediaPipe Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    detection_rate = (detected_count / frame_count * 100) if frame_count > 0 else 0
    print(f"\n✅ MediaPipe test completed!")
    print(f"Total frames: {frame_count}")
    print(f"Poses detected: {detected_count}")
    print(f"Detection rate: {detection_rate:.1f}%")
    
    return True

if __name__ == "__main__":
    test_mediapipe()
