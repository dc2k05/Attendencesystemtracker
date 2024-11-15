import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Load the face recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to train the recognizer using images
def train_recognizer(training_data_dir):
    face_samples = []
    ids = []
    label_names = {}
    
    # Loop through each person's folder and read their images
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
    
    # Train the recognizer
    recognizer.train(face_samples, np.array(ids))
    return label_names

# Function to mark attendance in the CSV file
def mark_attendance(name, recognized_names):
    file_path = 'student_attendance (2).csv'
    
    # Read the existing CSV file
    df = pd.read_csv(file_path)
    
    # Get the current time in 12-hour format with AM/PM
    current_time = datetime.now().strftime('%I:%M:%S %p')
    
    # Get the current day of the week (e.g., Monday, Tuesday)
    current_day = datetime.now().strftime('%A')
    
    # Update attendance for the recognized person if not already marked
    if name in df['Student Name'].values and name not in recognized_names:  # Ensure the student isn't already marked
        df.loc[df['Student Name'] == name, ['Attendance', 'Time', 'Day']] = ['Present', current_time, current_day]
        recognized_names.add(name)  # Add the student to the recognized set
    
    # Save the updated CSV
    df.to_csv(file_path, index=False)

# Function to mark absence for unrecognized individuals
def mark_absence(recognized_names):
    file_path = 'student_attendance (2).csv'
    
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Set the deadline time for attendance to 11:10 PM
    deadline = datetime.strptime('11:10 PM', '%I:%M %p')

    # Get the current time
    current_time = datetime.now()

    # Check if current time is past the deadline
    if current_time >= deadline:
        # Update attendance to 'Absent' for people who weren't recognized
        df.loc[~df['Student Name'].isin(recognized_names), 'Attendance'] = 'Absent'
    
    # Save the updated CSV
    df.to_csv(file_path, index=False)

# Main attendance system
def attendance_system(training_data_dir):
    # Train recognizer and get label names
    label_names = train_recognizer(training_data_dir)
    
    # Set to keep track of recognized names
    recognized_names = set()
    
    # Start webcam
    video_capture = cv2.VideoCapture(0)
    
    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)
            
            if confidence < 100:  # Threshold for face recognition confidence
                name = label_names[label]
                
                # Mark attendance only once per recognized person
                mark_attendance(name, recognized_names)
            else:
                name = "Unknown"
            
            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Display the video feed
        cv2.imshow('Video', frame)
        
        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Mark absence for people who weren't recognized after the deadline
    mark_absence(recognized_names)
    
    # Release webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()

# Example CSV initialization (to be run only once to create the file)
def create_initial_csv():
    # Data for the initial CSV
    data = {
        "Roll No.": [1, 2, 3, 4, 5, 6, 7, 46],
        "Student Name": [
            "ABISH S", "ARUN BALAJI R", "ANBU SELVAM N",
            "ASHWIN TAMIL SELVAN M.S", "DHARUN CHANDRU B",
            "ASHWANTH P", "THEOPHILUS F", "MAHESH ARAVIND M.S"
        ],
        "Attendance": [""] * 8,
        "Time": [""] * 8,
        "Day": [""] * 8
    }

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv('student_attendance (2).csv', index=False)

# Uncomment the next line to create the initial CSV file
create_initial_csv()

# Run the attendance system (replace with the directory containing known face images)
attendance_system('training_data')  # Ensure this directory contains folders for each student with their images
