import cv2
import numpy as np

from pitch import get_warped_pitch
from ultralytics import YOLO

cap = cv2.VideoCapture('video.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=25, detectShadows=False)
warped_pitch_image = get_warped_pitch()
# model = YOLO("yolo11s-pose.pt")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()


while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error reading frame.")
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    fg_mask = fgbg.apply(blurred_frame)
    _, white_mask = cv2.threshold(blurred_frame, 200, 255, cv2.THRESH_BINARY)

    moving_white_mask = cv2.bitwise_and(fg_mask, white_mask)
    moving_white_mask = cv2.morphologyEx(moving_white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(moving_white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)  # Green contours with thickness of 2

    moving_white_mask_3channel = cv2.merge([moving_white_mask] * 3)
    warped_pitch_image = cv2.resize(warped_pitch_image, (frame.shape[1], frame.shape[0]))
    overlay = cv2.addWeighted(moving_white_mask_3channel, 1, warped_pitch_image, 0.5, 0)
    
    cv2.imshow('Moving White Mask with Contours', overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
