import face_recognition
import cv2
import numpy as np
from datetime import datetime
import os
import csv

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# read all filenames
curr_dir = os.path.dirname(__file__)

images_path = os.path.join(curr_dir, 'images')

files = os.listdir(images_path)

images = [image for image in files if os.path.isfile(os.path.join(images_path, image))]


# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []
known_face_ids = []

# encode all images in images folder
for image in images:
    name = image.split('.')[0].capitalize()
    
    image_load = face_recognition.load_image_file(os.path.join(images_path, image))
    image_encoding = face_recognition.face_encodings(image_load)[0]
    
    name = name.split(' - ')
    name[1] = name[1].upper()
    
    known_face_encodings.append(image_encoding)
    known_face_names.append(name[0])
    known_face_ids.append(name[1])
    


student_presents = []

current_date = datetime.now().strftime('%Y-%m-%d')



# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

file_to_write = 'attendance/' + current_date + '.csv'

if os.path.isfile(file_to_write):
    # read file to load shits
    content = open(file_to_write,'r').readlines()
    student_presents = list(set(map(lambda x: x.split(',')[1], content)))

file = open(file_to_write, 'a+', newline='')
lnwriter = csv.writer(file)

while True:
    now = datetime.now()
    
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        code = cv2.COLOR_BGR2RGB
        rgb_small_frame = cv2.cvtColor(rgb_small_frame, code)
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_ids = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            id = ''
    
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                id = known_face_ids[best_match_index]

            face_names.append(name)
            face_ids.append(id)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name, id in zip(face_locations, face_names, face_ids):
        
        color = (0, 0,255)
        if name in student_presents:
            color = (0, 255, 0)
        
            
        
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        # Draw id rectangle
        cv2.rectangle(frame, (left, bottom + 35), (right, bottom), color, cv2.FILLED)
        # draw the id
        cv2.putText(frame, id, (left + 6, bottom + 25), font, .6, (255,255,255), 1)
        
        if name in known_face_names:
            if name not in student_presents:
                lnwriter.writerow([id, name, now])
                student_presents.append(name)

            
        
    
    # display datetime
    date = now.strftime('%Y-%m-%d %H:%M:%S')
    
    cv2.rectangle(frame, (0,0), (350, 30), (0, 0, 255), cv2.FILLED)
    
    font = cv2.FONT_HERSHEY_TRIPLEX 
    cv2.putText(frame, date, (5, 20), font, .8, (255, 255, 255), 1)

    

    # Display the resulting image
    cv2.imshow('Attendace system', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()