import os
import cv2
import mediapipe as mp


# Creation of the folder where we will store the photos
Addres = 'C:/Users/matia/Desktop/FaceMask-Detection/images'
name = 'Matt_FaceMask'
folder = Addres + '/' + name

if not os.path.exists(folder):
    print('Folder created.')
    os.makedirs(folder)

# Detector

detector = mp.solutions.face_detection 
draw = mp.solutions.drawing_utils 

# Video capture

cap = cv2.VideoCapture(1)


# Now initialize the detection parameters
with detector.FaceDetection(min_detection_confidence=0.75) as faces:
    while True: 

        ret, frame = cap.read()

        # Remove the parallax effect
        frame = cv2.flip(frame, 1)

        # RGB color correction.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection 
        consequence = faces.process(rgb)   

        #Security filter
        if consequence.detections is not None:
            for face in consequence.detections:
                draw.draw_detection(frame, face, draw.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))
                print(face)

        #Displaying the frames and assigning a key

        cv2.imshow("Facial recognition with face masks", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break


cap.release()
cv2.destroyAllWindows()

