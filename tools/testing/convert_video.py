import cv2
import sys

input_path = 'data/raw/videos/benchpress/incorrect/in_1.mov'
output_path = 'data/raw/videos/benchpress/incorrect/in_1_web.webm'

print(f"Opening {input_path}")
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Cannot open video")
    sys.exit()

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video is {w}x{h} @ {fps}fps, {total} frames")

fourcc = cv2.VideoWriter_fourcc(*'vp09') # VP9 support
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

if not out.isOpened():
    print("Cannot open writer with vp09, trying VP80")
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

c = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    c += 1
    if c % 100 == 0:
        print(f"Converted {c}/{total} frames...")

cap.release()
out.release()
print(f"Finished writing {c} frames to {output_path}")
