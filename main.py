import os
import glob
from utils import extract_frames, encode_known_faces, recognize_multiple_faces
import multiprocessing

# Paths
VIDEO_FOLDER = "videos"
PHOTO_FOLDER = "student_photos"
OUTPUT_FOLDER = "output"

# User Input
video_file = input("Enter video file name (inside videos/ folder): ")
video_path = os.path.join(VIDEO_FOLDER, video_file)
frame_folder = os.path.join(OUTPUT_FOLDER, "frames")

# Step 1: Extract frames
print("Extracting frames from video...")
frames_to_extract = 50
extract_frames(video_path, frame_folder, frames_to_extract)

# Step 2: Encode known student faces
print("Encoding student photos from folder...")
known_encodings = encode_known_faces(PHOTO_FOLDER)

# Step 3: Recognize faces using multiprocessing
frame_paths = glob.glob(os.path.join(frame_folder, "*.jpg"))

def process_frame(args):
    known_encodings, frame_path = args
    return recognize_multiple_faces(known_encodings, frame_path)

print("Recognizing faces using multiprocessing...")
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    results = pool.map(process_frame, [(known_encodings, path) for path in frame_paths])

# Step 4: Collect results
all_recognized = set()
for result in results:
    all_recognized.update(result)

# Step 5: Print attendance
print("Students present in video:")
for student in sorted(all_recognized):
    print(f"- {student}")
