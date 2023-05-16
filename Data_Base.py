import os
import cv2
import mediapipe as mp


# Creation of the folder where we will store the photos
Addres = 'C:/Users/matia/Desktop/FaceMask-Detection/images'
name = 'Matt_Face1'
folder = Addres + '/' + name

if not os.path.exists(folder):
    print('Folder created.')
    os.makedirs(folder)

cont = 0 #contador

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

        # Security filter
        if consequence.detections is not None:
            for face in consequence.detections:
                #draw.draw_detection(frame, face, draw.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))
                print(face)

                he, wi, nm = frame.shape

                # Extracting initial X, initial Y and width, height.
                xi = face.location_data.relative_bounding_box.xmin
                yi = face.location_data.relative_bounding_box.ymin
 
                width = face.location_data.relative_bounding_box.width
                height = face.location_data.relative_bounding_box.height

                # Conversion to pixels.

                xi = int(xi * wi)
                yi = int(yi * he)
                width = int(width * wi)
                height = int(height * he)

                #We find the final X and final Y.

                xf = xi + width
                yf = yi + height

                # Pixel extraction.

                face = frame[yi:yf, xi:xf]

                # Resizing the images and storing our images.

                face = cv2.resize(face,(200, 250), interpolation=cv2.INTER_CUBIC)

                cv2.imwrite(folder + '/face_{}.jpg'.format(cont), face)
                cont = cont + 1


        # Displaying the frames and assigning a key

        cv2.imshow("Facial recognition with face masks", frame)
        k = cv2.waitKey(1)
        if k == 27 or cont > 300:
            break


cap.release()
cv2.destroyAllWindows()

