import cv2
import mediapipe as mp
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Play video with MediaPipe Pose overlay. Press SPACE to pause, ESC to quit.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    args = parser.parse_args()

    if not args.video:
        print("Please provide a video path. Example: python play_with_mediapipe.py --video data/raw/videos/benchpress/correct/corr_01.mp4")
        sys.exit(1)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video file: {args.video}")
        sys.exit(1)

    print("\n" + "="*50)
    print("Controls:")
    print("  SPACE : Pause / Resume (You can screenshot when paused)")
    print("  ESC   : Quit")
    print("="*50 + "\n")

    paused = False

    cv2.namedWindow('FitVision - MediaPipe Overlay', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('FitVision - MediaPipe Overlay', 800, 600)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Reached end of video.")
                    break
                
                # Get pose
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                annotated_frame = frame.copy()
                if results.pose_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                cv2.imshow('FitVision - MediaPipe Overlay', annotated_frame)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27: # ESC
                break
            elif key == ord(' '): # SPACE
                paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
