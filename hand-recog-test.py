# Install necessary libraries (only run this in a Jupyter Notebook or similar environment)
# !pip install mediapipe opencv-python

# Import dependencies
import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

# Initialize MediaPipe hand drawing and hand detection solutions
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Create an output directory if it doesn't exist
output_dir = 'Output Images'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Function to draw hands and save images
def draw_hands_and_save_images():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Flip horizontally for a mirror-view display
            image = cv2.flip(image, 1)

            # Improve performance by marking the image as not writeable
            image.flags.writeable = False

            # Process the image and detect hands
            results = hands.process(image)

            # Mark the image as writeable again
            image.flags.writeable = True

            # Convert RGB to BGR for OpenCV display
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS, 
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )

            # Save the image to the output directory
            cv2.imwrite(os.path.join(output_dir, f'{uuid.uuid1()}.jpg'), image)

            # Display the frame with hand landmarks
            cv2.imshow('Hand Tracking', image)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the function
draw_hands_and_save_images()
