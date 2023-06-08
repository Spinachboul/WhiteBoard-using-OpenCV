import cv2
import numpy as np
import mediapipe as mp

// Code By Mridul Jain

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Create a whiteboard canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8) + 255

# Initialize variables for drawing
drawing = False
prev_point = None

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the position of the index fingertip
            index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            fingertip_x = int(index_fingertip.x * frame.shape[1])
            fingertip_y = int(index_fingertip.y * frame.shape[0])

            # Draw a circle at the fingertip position
            cv2.circle(frame, (fingertip_x, fingertip_y), 5, (0, 0, 255), -1)

            # Start drawing if index finger is up
            if index_fingertip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y:
                if prev_point is None:
                    prev_point = (fingertip_x, fingertip_y)
                else:
                    cv2.line(canvas, prev_point, (fingertip_x, fingertip_y), (0, 0, 0), 2)
                    prev_point = (fingertip_x, fingertip_y)
                    drawing = True
            else:
                prev_point = None
                drawing = False

    # Detect contours on the canvas
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_canvas, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Display the frame and canvas
    cv2.imshow("Hand Object Detection", frame)
    cv2.imshow("Whiteboard", canvas)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
