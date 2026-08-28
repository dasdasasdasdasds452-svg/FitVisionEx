"""
Test Complete Pipeline: YOLOv8 + MediaPipe
"""
import cv2
import mediapipe as mp
from ultralytics import YOLO

def test_pipeline():
    print("Testing Complete Pipeline (YOLOv8 + MediaPipe)...")
    
    # Initialize
    print("Loading models...")
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    yolo = YOLO('yolov8s.pt')
    print("✅ Models loaded successfully")
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return False
    
    print("✅ Webcam opened successfully")
    print("Press 'q' to quit")
    
    frame_count = 0
    pipeline_success_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        display_frame = frame.copy()
        
        # Step 1: Detect person with YOLO
        results = yolo(frame, classes=[0], verbose=False)
        
        if len(results[0].boxes) > 0:
            # Get first person
            box = results[0].boxes[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, "Person", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Crop person
            person_img = frame[y1:y2, x1:x2]
            
            if person_img.size > 0:
                # Step 2: Pose estimation
                rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb_img)
                
                if pose_results.pose_landmarks:
                    pipeline_success_count += 1
                    
                    # Draw landmarks on cropped image
                    mp_drawing.draw_landmarks(
                        person_img,
                        pose_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    # Put back to frame
                    display_frame[y1:y2, x1:x2] = person_img
                    
                    cv2.putText(display_frame, "✓ Pose Detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "✗ No Pose", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "No Person Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display stats
        success_rate = (pipeline_success_count / frame_count * 100) if frame_count > 0 else 0
        cv2.putText(display_frame, f"Frame: {frame_count} | Success: {pipeline_success_count} ({success_rate:.1f}%)", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Pipeline Test', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    success_rate = (pipeline_success_count / frame_count * 100) if frame_count > 0 else 0
    print(f"\n✅ Pipeline test completed!")
    print(f"Total frames: {frame_count}")
    print(f"Successful detections: {pipeline_success_count}")
    print(f"Success rate: {success_rate:.1f}%")
    
    return True

if __name__ == "__main__":
    test_pipeline()
