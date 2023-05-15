# Import the libraries
import cv2
import mediapipe as mp

# Declare detector
detector = mp.solutions.face_detection  # Detector
draw_tool = mp.solutions.drawing_utils  # Draw

# Perform video capture
capture = cv2.VideoCapture(1)

# Initialize the parameters
with detector.FaceDetection(min_detection_confidence=0.75) as face_detection:
    while True:
        # We perform video capture reading
        ret, frame = capture.read()   

        # Remove the parallax effect
        frame = cv2.flip(frame, 1)    

        # RGB color correction.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection
        consequence = face_detection.process(rgb)


        # Filter
        if consequence.detections is not None:
            for detection in consequence.detections:
                draw_tool.draw_detection(frame, detection, draw_tool.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

                for id, coordinates in enumerate(consequence.detections):
                    #Show coords
                    #print("coordinates: ", consequence.detections)

                    # Extraction of dimensions from our image.
                    
                    hi, an, c = frame.shape

                    # Extract initial X and Y  
                    x = coordinates.location_data.relative_bounding_box.xmin
                    y = coordinates.location_data.relative_bounding_box.ymin

                    # Extract Width and Height 

                    wth = coordinates.location_data.relative_bounding_box.width
                    hgt = coordinates.location_data.relative_bounding_box.height

                    #Conversion to pixels 
                    x_i, y_i = int(x * an), int(y * hi)
                    x_f, y_f = int(wth * an), int(hgt * hi)

                    #Central point in face general area

                    coord_x = (x_i + (x_i + x_f )) // 2
                    coord_y = (y_i +(y_i + y_f)) // 2

                    #Show the coords
                    
                    cv2.circle(frame, (coord_x, coord_y), 4, (255, 100, 0), cv2.FILLED)


        # We display the frames and read the keyboard
        cv2.imshow('Camera-FaceDetection', frame) 
        e = cv2.waitKey(1)
        if e == 27:
            break

capture.release()
cv2.destroyAllWindows()
