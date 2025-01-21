import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('video.mp4')

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

def mask_white(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the grayscale frame
    blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)

    # Threshold the grayscale frame to isolate white areas
    _, mask = cv2.threshold(blurred_frame, 200, 220, cv2.THRESH_BINARY)

    # Create a three-channel version of the mask to apply to the original frame
    mask_3channel = cv2.merge([mask, mask, mask])

    # Apply the mask to the original frame
    return cv2.bitwise_and(frame, mask_3channel)


if __name__ == "__main__":
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            print("End of video or error reading frame.")
            break

        masked_frame = mask_white(frame)

        # Display the masked frame
        cv2.imshow('Masked Video', masked_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()