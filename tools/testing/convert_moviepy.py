from moviepy import VideoFileClip
import sys

input_path = 'data/raw/videos/benchpress/incorrect/in_1.mov'
output_path = 'data/raw/videos/benchpress/incorrect/in_1_web.mp4'

try:
    print(f"Loading '{input_path}'...")
    clip = VideoFileClip(input_path)
    
    # Optional: resize if video is too large to speed up processing
    if clip.w > 1280:
        clip = clip.resize(width=1280)
        
    print(f"Converting to '{output_path}' with H.264 codec...")
    clip.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )
    print("Conversion successful!")

except Exception as e:
    print(f"Error during conversion: {e}")
    sys.exit(1)
