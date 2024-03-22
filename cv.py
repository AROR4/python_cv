
import face_recognition
import numpy as np
import cv2
cap=cv2.VideoCapture(0)
import cvzone
from cvzone.HandTrackingModule import HandDetector
qcd = cv2.QRCodeDetector()

raghav_image = face_recognition.load_image_file("raghav.jpg")
raghav_face_encoding = face_recognition.face_encodings(raghav_image)[0]

known_face_encodings = [
    raghav_face_encoding
]
known_face_names = [
    "raghav arora"
]
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
detector=HandDetector(detectionCon=0.8,maxHands=2)
while True:
    s,img=cap.read()
    hands, img = detector.findHands(img)
    retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(img)
    print(decoded_info)
    if decoded_info and len(decoded_info[0])>0:
        cv2.destroyAllWindows()
        break
    cv2.imshow("image window",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

while True:
    
    # Capture frame-by-frame
    ret, frame_o = cap.read()

    if process_this_frame:

        frame = cv2.resize(frame_o, (0, 0), fx=0.25, fy=0.25)

    # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare the face with the known face of Raghav
            matches = face_recognition.compare_faces([raghav_face_encoding], face_encoding)

            name = "Unknown"
            if matches[0]:
                name = "Raghav"

            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame_o, (left*4, top*4), (right*4, bottom*4), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_o, name, (left*4 + 6, bottom*4 - 6), font, 0.5, (255, 255, 255), 1)
        # Display the resulting frame
        cv2.imshow('Video', frame_o)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    process_this_frame = not process_this_frame
