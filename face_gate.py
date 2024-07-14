import face_recognition # AI
import time # Time from processor clock
from imutils import paths # to read folders & images
import os # to read folders & images
import cv2 # OpenCV for image processing
import numpy as np # for image processing

# Initialize some variables for both departments
known_face_names = [] # to store names in dataset
known_face_ids = [] # to store face ids in dataset
dataset_path = "dataset"

## START --- load dataset ---
for person_name in os.listdir(dataset_path): # read folders [Raghad, Remas]
    # read files one by one
    # dataset/Raghad
    person_folder = os.path.join(dataset_path, person_name)

    # Check if it's a directory
    if os.path.isdir(person_folder):
        # Get all image paths for the current person
        image_paths = list(paths.list_images(person_folder))
        # loop over the image paths
        for image_path in image_paths: # read images
            # extract the person name from the image path
            print("[INFO] processing images")
            # name = image_path.split(os.path.sep)[-2]
            ######################################################################
            # Load a sample picture and learn how to recognize it.
            image = face_recognition.load_image_file(image_path)
            #get face Id
            face_encodings = face_recognition.face_encodings(image)
            ######################################################################   
            # In some cases, there might be no face found in an image
            # len = length
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                # store person name in known_face_names variable
                known_face_names.append(person_name)
                # store person face id in known_face_ids variable
                known_face_ids.append(face_encoding)
                print(f"Encoded face for {person_name}")
            else:
                print(f"No face found in {image_path}")
## END --- load dataset ---

camera = cv2.VideoCapture(0)
while True: # infinite loop
    # Grab a single frame of video
    _, frame = camera.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_small_frame = small_frame[:, :, ::-1]
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    # 1. Face detection
    # 2. Face recognition
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

    # Display
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
camera.release()
cv2.destroyAllWindows()