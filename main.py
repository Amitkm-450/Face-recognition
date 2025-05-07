import os
import glob
import multiprocessing
from utils import (
    extract_frames,
    encode_known_faces,
    recognize_multiple_faces,
    download_student_photos,
    get_student_image_urls,
)

# Paths
VIDEO_FOLDER = "videos"
PHOTO_FOLDER = "student_photos"
OUTPUT_FOLDER = "output"

print("Fetching student image URLs from MongoDB...")
urls = get_student_image_urls()

print("Downloading student photos from S3 signed URLs...")
download_student_photos(urls, download_folder=PHOTO_FOLDER)

video_file = input("Enter video file name (inside videos/ folder): ")
video_path = os.path.join(VIDEO_FOLDER, video_file)
frame_folder = os.path.join(OUTPUT_FOLDER, "frames")

print("Extracting frames from video...")
extract_frames(video_path, frame_folder, num_frames=50)

# Encode student images
print("Encoding student photos...")
known_encodings = encode_known_faces(PHOTO_FOLDER)

# Setup multiprocessing to recognize faces
frame_paths = glob.glob(os.path.join(frame_folder, "*.jpg"))

# -- Multiprocessing setup --
global_encodings = None  # To be set in worker initializer

def init_worker(encodings):
    global global_encodings
    global_encodings = encodings

def process_frame(frame_path):
    return recognize_multiple_faces(global_encodings, frame_path)

print("Recognizing faces using multiprocessing...")
with multiprocessing.Pool(
    processes=multiprocessing.cpu_count(),
    initializer=init_worker,
    initargs=(known_encodings,)
) as pool:
    results = pool.map(process_frame, frame_paths)

# Collect results and show attendance
all_recognized = set()
for result in results:
    all_recognized.update(result)

print("\n🎓 Students present in the video:")
for student in sorted(all_recognized):
    print(f" - {student}")
