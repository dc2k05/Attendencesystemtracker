import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Load the face recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set initial file path for attendance.csv in the working directory
file_path = 'attendance tracker.csv'

# Function to train the recognizer using images
def train_recognizer(training_data_dir):
    face_samples = []
    ids = []
    label_names = {}

    for idx, person in enumerate(os.listdir(training_data_dir)):
        label_names[idx] = person
        person_folder = os.path.join(training_data_dir, person)

        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

            for (x, y, w, h) in faces:
                face_samples.append(gray_img[y:y+h, x:x+w])
                ids.append(idx)

    recognizer.train(face_samples, np.array(ids))
    return label_names

# Function to mark attendance in the CSV file
def mark_attendance(name, recognized_names):
    try:
        df = pd.read_csv(file_path)
        current_time = datetime.now().strftime('%I:%M:%S %p')
        current_day = datetime.now().strftime('%A')

        # Check if the student name exists in the CSV and hasn't been marked already
        if name in df['Student_name'].values and name not in recognized_names:
            df.loc[df['Student_name'] == name, ['Attendance', 'Time', 'Day']] = ['Present', current_time, current_day]
            recognized_names.add(name)
        df.to_csv(file_path, index=False)
    except PermissionError as e:
        print(f"PermissionError: {e}. Please close any other program accessing {file_path}.")
    except Exception as e:
        print(f"An error occurred while marking attendance: {e}")

# Function to mark absence for unrecognized individuals after the deadline and save to Downloads
def mark_absence_and_move(recognized_names):
    try:
        df = pd.read_csv(file_path)
        deadline = datetime.strptime('03:36 PM', '%I:%M %p')  # Adjusted to 3:36 PM
        current_time = datetime.now()

        # Update attendance to 'Absent' for people who weren't recognized after the deadline
        if current_time >= deadline:
            df.loc[~df['Student_name'].isin(recognized_names), 'Attendance'] = 'Absent'
            df.to_csv(file_path, index=False)

            # Move the file to the Downloads folder after marking absences
            downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
            new_file_path = os.path.join(downloads_folder, 'attendance tracker.csv')
            os.replace(file_path, new_file_path)
            print(f"Attendance file moved to: {new_file_path}")

    except PermissionError as e:
        print(f"PermissionError: {e}. Please close any other program accessing {file_path}.")
    except Exception as e:
        print(f"An error occurred while marking absence: {e}")

# Main attendance system
def attendance_system(training_data_dir):
    label_names = train_recognizer(training_data_dir)
    recognized_names = set()
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        ret, frame = video_capture.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)

            if confidence < 100:
                name = label_names[label]
                mark_attendance(name, recognized_names)
            else:
                name = "Unknown"

            # Draw bounding box and display the name on the video feed
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Mark absence and move file after deadline
    mark_absence_and_move(recognized_names)
    video_capture.release()
    cv2.destroyAllWindows()

# Example CSV initialization (to be run only once to create the file)
def create_initial_csv():
    data = {
        "Roll_no": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Student_name": [
            "ABDUL WAHID Z", "ANBU SELVAM N", "ARUN BALAJI R", "ASHWIN TAMIL SELVAN M.S", "BHARATH P",
            "DHANUSH S", "DHARUN CHANDRU B", "MAHESH ARAVIND M.S", "VENKATESH PERUMAL", "VICE PRINCIPAL"
        ],
        "Attendance": [""] * 10,
        "Time": [""] * 10,
        "Day": [""] * 10
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

# Uncomment the line below if you need to create the initial CSV
# create_initial_csv()

attendance_system('training_data')  # Ensure the 'training_data' folder exists with images of each student
