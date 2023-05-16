#Imports
import os
import cv2
import mediapipe as mp

addres = 'C:\\Users\\matia\\Desktop\\FaceMask-Detection\\images'
Labels = os.listdir(addres)
cont = 0 #contador

print('Names:', Labels)

#Calling the trained model

model = cv2.face.LBPHFaceRecognizer_create()

# Model

model.read('TrainModel.xml')

# Detector

detector = mp.solutions.face_detection 
draw = mp.solutions.drawing_utils 

# Video capture

cap = cv2.VideoCapture(1)


# Now initialize the detection parameters
with detector.FaceDetection(min_detection_confidence=0.75) as faces:
    while True: 

        ret, frame = cap.read()
        copy = frame.copy()

        # Remove the parallax effect
        frame = cv2.flip(copy, 1)

        # RGB color correction.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); copy2 = rgb.copy()
        
        # Face detection 
        consequence = faces.process(copy2)   

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

                face = copy2[yi:yf, xi:xf]

                # Resizing the images.

                face = cv2.resize(face,(200, 250), interpolation=cv2.INTER_CUBIC)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # Prediction

                prediction = model.predict(face)

                #Show Results

                if prediction[0] == 0:
                    cv2.putText(frame, '{}'.format(Labels[0]), (xi, yi - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (xi, yi), (xf, yf), (0, 0, 255), 2)
                elif prediction[0] == 1:  # Cambia esto a 1 en lugar de 0
                    cv2.putText(frame, '{}'.format(Labels[1]), (xi, yi - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xi, yi), (xf, yf), (255, 0, 0), 2)


        # Displaying the frames and assigning a key

        cv2.imshow("Facial recognition with face masks", frame)
        k = cv2.waitKey(1)
        if k == 27 or cont > 300:
            break


cap.release()
cv2.destroyAllWindows()

