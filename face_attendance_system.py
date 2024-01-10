import face_recognition
import openpyxl
import xlrd
from datetime import date
from datetime import datetime
import time
from imutils import paths
import os
import cv2
import numpy as np

# Initialize some variables for both departments
known_face_names = []
known_face_ids = []
dataset_path = "dataset"

# Dictionary to keep track of sign-ins
last_attend_times = {}
last_leave_times = {}

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    # Check if it's a directory
    if os.path.isdir(person_folder):
        # Get all image paths for the current person
        image_paths = list(paths.list_images(person_folder))
        # loop over the image paths
        for image_path in image_paths:
            # extract the person name from the image path
            print("[INFO] processing images")
            # name = image_path.split(os.path.sep)[-2]

            # Load a sample picture and learn how to recognize it.
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
                
            # In some cases, there might be no face found in an image
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                known_face_names.append(person_name)
                known_face_ids.append(face_encoding)
                print(f"Encoded face for {person_name}")
            else:
                print(f"No face found in {image_path}")


def getFrameAfterRecognition(frame):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_small_frame = small_frame[:, :, ::-1]
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Only process every other frame of video to save time
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_ids, face_encoding)
        name = "Unknown"
    
        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_ids, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if name == 'Unknown':
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame, face_names

def attend(person):
    j=0
    while True:
        if person == "Unknown":
            break
        attendace_file  = "csv/attendace_report.xlsx"
        attendace_wb = openpyxl.load_workbook(filename=attendace_file)
        attendace_sheet = attendace_wb.active
        attendace_read_wb = xlrd.open_workbook(attendace_file)
        attendace_read_sheet = attendace_read_wb.sheet_by_index(0)
        try:
            Date = str(attendace_read_sheet.cell_value(j, 0))
        except IndexError:
            j=j+1
            attendace_sheet["A"+str(j)] = person
            attendace_sheet["B"+str(j)] = "Entered"
            attendace_sheet["C"+str(j)] = date.today()
            attendace_sheet["D"+str(j)] = datetime.now().strftime("%H:%M:%S")
            try:
                attendace_wb.save(filename=attendace_file)
            except IOError:
                print(attendace_file+" is open, please close file & try again!")
    
            break
        j=j+1

def left(person):
    j=0
    while True:
        if person == "Unknown":
            break
        attendace_file  = "csv/attendace_report.xlsx"
        attendace_wb = openpyxl.load_workbook(filename=attendace_file)
        attendace_sheet = attendace_wb.active
        attendace_read_wb = xlrd.open_workbook(attendace_file)
        attendace_read_sheet = attendace_read_wb.sheet_by_index(0)
        try:
            Date = str(attendace_read_sheet.cell_value(j, 0))
        except IndexError:
            j=j+1
            attendace_sheet["A"+str(j)] = person
            attendace_sheet["B"+str(j)] = "Left"
            attendace_sheet["C"+str(j)] = date.today()
            attendace_sheet["D"+str(j)] = datetime.now().strftime("%H:%M:%S")
            try:
                attendace_wb.save(filename=attendace_file)
            except IOError:
                print(attendace_file+" is open, please close file & try again!")
    
            break
        j=j+1

def can_attend(person):
    current_time = time.time()
    if person in last_attend_times:
        last_attend_time = last_attend_times[person]
        # Check if 30 seconds have passed since last sign-in
        if current_time - last_attend_time < 30:
            return False
    # Update the sign-in time
    last_attend_times[person] = current_time
    return True

def can_leave(person):
    current_time = time.time()
    if person in last_leave_times:
        last_leave_time = last_leave_times[person]
        # Check if 30 seconds have passed since last sign-in
        if current_time - last_leave_time < 30:
            return False
    # Update the sign-in time
    last_leave_times[person] = current_time
    return True

if __name__ == "__main__":
    entrance_camera = cv2.VideoCapture(0)
    exit_camera = cv2.VideoCapture(1)
    while True:
        # Grab a single frame of video
        _, entrance_frame = entrance_camera.read()
        _, exit_frame = exit_camera.read()

        entrance_frame, enter_people = getFrameAfterRecognition(entrance_frame)
        exit_frame, exit_people = getFrameAfterRecognition(exit_frame)

        for person in enter_people:
            if can_attend(person):
                attend(person)
                print(f"Attended: {person}")
        
        for person in exit_people:
            if can_leave(person):
                left(person)
                print(f"Left: {person}")

        cv2.imshow('Entrance Video', entrance_frame)
        cv2.imshow('Exit Video', exit_frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    entrance_camera.release()
    # exit_camera.release()
    cv2.destroyAllWindows()