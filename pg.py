import cv2
import numpy as np
from ultralytics import YOLO
from pitch import get_warped_pitch
from pose_estimation import draw_skeleton

cap = cv2.VideoCapture('video.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=25, detectShadows=False)
warped_pitch_image = get_warped_pitch()
model = YOLO("yolo11s-pose.pt")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties for output file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('results.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Initialize trajectory lists for ball and shoes
ball_trajectory = []
left_shoe_trajectory = []
right_shoe_trajectory = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error reading frame.")
        break

    results = model.track(frame, persist=True)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    _, white_mask = cv2.threshold(blurred_frame, 200, 255, cv2.THRESH_BINARY)
    fg_mask = fgbg.apply(blurred_frame)

    moving_white_mask = cv2.bitwise_and(fg_mask, white_mask)
    moving_white_mask = cv2.morphologyEx(moving_white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    moving_white_mask_3channel = cv2.merge([moving_white_mask] * 3)
    warped_pitch_image = cv2.resize(warped_pitch_image, (frame.shape[1], frame.shape[0]))
    overlay = cv2.addWeighted(moving_white_mask_3channel, 1, warped_pitch_image, 0.5, 0)

    # Detect contours in the moving white mask
    contours, _ = cv2.findContours(moving_white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Handle pose estimation overlay
    if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
        keypoints = results[0].keypoints.data.cpu().numpy()
        draw_skeleton(overlay, keypoints)

    # Write the frame to the output video file
    out.write(overlay)

    # Display the frame
    cv2.imshow('Moving White Mask with Contours', overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Save the video file
cv2.destroyAllWindows()