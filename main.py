import os
import glob
from utils import extract_frames, encode_face, recognize_faces_optimized
import multiprocessing

# Paths
VIDEO_FOLDER = "videos"
PHOTO_FOLDER = "student_photos"
OUTPUT_FOLDER = "output"

# User Inputs
video_file = input("Enter video file name (inside videos/ folder): ")
student_photo = input("Enter student photo file name (inside student_photos/ folder): ")

video_path = os.path.join(VIDEO_FOLDER, video_file)
photo_path = os.path.join(PHOTO_FOLDER, student_photo)
frame_folder = os.path.join(OUTPUT_FOLDER, "frames")

# Extract frames from the video - Optimized extraction
print("Extracting frames from video...")
# Only extract a subset of frames
frames_to_extract = 50  # Adjust number as needed for balance of speed and accuracy
extract_frames(video_path, frame_folder, frames_to_extract)
#frames = extract_frames(video_path, frame_folder) #removed the frame return for memory optimisation

# Encode the student's photo
print("Encoding student photo...")
student_encoding = encode_face(photo_path)

if student_encoding is None:
    print("Error: No face detected in student photo.")
else:
    # Recognize faces in video frames - using optimized function and multiprocessing
    print("Matching student photo with video frames...")

    # Get all frame paths for parallelization
    frame_paths = glob.glob(os.path.join(frame_folder, "*.jpg"))
    
    # Multiprocessing setup
    num_processes = multiprocessing.cpu_count()  # Use all available cores
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Use the optimized recognition function
    results = pool.starmap(recognize_faces_optimized, [(student_encoding, frame_path) for frame_path in frame_paths])
    
    pool.close()
    pool.join()
    
    # Check if the student is present in any frame
    is_present = any(results)

    # Mark attendance
    if is_present:
        print(f"{student_photo} is Present in the class.")
    else:
        print(f"{student_photo} is Absent from the class.")
