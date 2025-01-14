"""
Make frames seperated at timesteps for testing purposes.
"""

import os
import cv2


# create output folder if it doesn't exist
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# load video
video_path = "./test_pipeline/pov.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("error: cannot open video file")
    exit()

# get fps and calculate frame interval
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 5)

frame_count = 0
saved_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # save frame every 5 seconds
    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(
            output_folder, f"frame_{saved_frame_count:04d}.jpg"
        )
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

    frame_count += 1

cap.release()
print(f"extraction complete. total frames saved: {saved_frame_count}")
