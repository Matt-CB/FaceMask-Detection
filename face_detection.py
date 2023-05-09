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
                draw_tool.draw_detection(frame, detection)

        # We display the frames and read the keyboard
        cv2.imshow('Camera-FaceDetection', frame)
        e = cv2.waitKey(1)
        if e == 27:
            break

capture.release()
cv2.destroyAllWindows()