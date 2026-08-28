"""
Test YOLOv5 Person Detection
"""
from ultralytics import YOLO
import cv2

def test_yolo():
    print("Testing YOLOv8 Person Detection...")
    
    # Load pre-trained YOLOv8 model
    print("Loading YOLOv8s model...")
    model = YOLO('yolov8s.pt')
    print("✅ Model loaded successfully")
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return False
    
    print("✅ Webcam opened successfully")
    print("Press 'q' to quit")
    
    frame_count = 0
    person_detected_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect only persons (class 0)
        results = model(frame, classes=[0], verbose=False)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        num_persons = len(results[0].boxes)
        if num_persons > 0:
            person_detected_count += 1
        
        cv2.putText(annotated_frame, f"Persons detected: {num_persons}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('YOLOv8 Test', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    detection_rate = (person_detected_count / frame_count * 100) if frame_count > 0 else 0
    print(f"\n✅ YOLOv8 test completed!")
    print(f"Total frames: {frame_count}")
    print(f"Frames with person: {person_detected_count}")
    print(f"Detection rate: {detection_rate:.1f}%")
    
    return True

if __name__ == "__main__":
    test_yolo()
