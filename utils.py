import cv2
import face_recognition
import os
import requests
import glob
from pymongo import MongoClient

def extract_frames(video_path, output_folder, num_frames=50):
    """
    Extracts a specified number of frames from a video file.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = total_frames // num_frames if num_frames < total_frames else 1
    
    extracted_count = 0
    for i in range(0, total_frames, frame_interval):
        if extracted_count >= num_frames:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if ret:
            frame_path = os.path.join(output_folder, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_count +=1
        else:
            print(f"Error: could not read frame number {i}")

    cap.release()

def encode_known_faces(known_faces_folder):
    """
    Encodes all faces in the given folder.
    Returns a dictionary: {student_name: face_encoding}
    """
    encodings = {}
    image_paths = glob.glob(os.path.join(known_faces_folder, "*.jpg"))

    for image_path in image_paths:
        name = os.path.splitext(os.path.basename(image_path))[0]
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                encodings[name] = face_encodings[0]
            else:
                print(f"No face found in {image_path}")
        except Exception as e:
            print(f"Error encoding {image_path}: {e}")

    return encodings


def recognize_multiple_faces(known_encodings, frame_path):
    """
    Recognizes multiple known faces in a single frame.
    Returns a list of recognized student names.
    """
    recognized_students = []

    try:
        frame = face_recognition.load_image_file(frame_path)
        frame_encodings = face_recognition.face_encodings(frame)

        for frame_encoding in frame_encodings:
            matches = face_recognition.compare_faces(list(known_encodings.values()), frame_encoding)
            for idx, match in enumerate(matches):
                if match:
                    student_name = list(known_encodings.keys())[idx]
                    if student_name not in recognized_students:
                        recognized_students.append(student_name)
        return recognized_students
    except Exception as e:
        print(f"Error recognizing faces in {frame_path}: {e}")
        return []

def get_student_image_urls():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["your_db_name"]
    students = db.students.find()
    
    student_images = {}
    for student in students:
        if "studentID" in student and "image" in student:
            student_images[student["studentID"]] = student["image"]  # image = signed URL
    return student_images

def download_student_photos(image_urls, download_folder="student_photos"):
    os.makedirs(download_folder, exist_ok=True)
    for student_id, url in image_urls.items():
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join(download_folder, f"{student_id}.jpg"), "wb") as f:
                    f.write(response.content)
                print(f"Downloaded: {student_id}.jpg")
            else:
                print(f"Failed to download image for {student_id}: {response.status_code}")
        except Exception as e:
            print(f"Error downloading image for {student_id}: {e}")