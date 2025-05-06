import cv2
import face_recognition
import os
import glob

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

def encode_face(image_path):
    """
    Encodes a face from an image.
    """
    try:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            return face_encodings[0]
        else:
            return None
    except Exception as e:
        print(f"Error encoding face: {e}")
        return None

def recognize_faces_optimized(student_encoding, frame_path):
    """
    Recognizes faces in a single frame.
    """
    try:
        frame = face_recognition.load_image_file(frame_path)
        frame_encodings = face_recognition.face_encodings(frame)

        for frame_encoding in frame_encodings:
            matches = face_recognition.compare_faces([student_encoding], frame_encoding)
            if True in matches:
                return True  # Student found in this frame
        return False  # Student not found in this frame
    except Exception as e:
        print(f"Error recognizing faces in {frame_path}: {e}")
        return False
