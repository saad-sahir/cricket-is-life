import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11s-pose.pt")
cap = cv2.VideoCapture('video.mp4')

# Define a function to draw the skeleton
def draw_skeleton(frame, keypoints):
    # Define connections using keypoint indices
    connections = [
        # Head and Face
        (0, 1), (0, 2), (1, 3), (2, 4),
        # Upper Body
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        # Lower Body
        (5, 11), (6, 12), (11, 12), (11, 13), (12, 14),
        (13, 15), (14, 16)
    ]

    for person in keypoints:
        try:
            # Draw keypoints
            for keypoint in person:
                x, y, conf = keypoint
                if conf > 0.5:  # Only draw keypoints with sufficient confidence
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            # Draw connections
            for pt1, pt2 in connections:
                if person[pt1, 2] > 0.5 and person[pt2, 2] > 0.5:  # Check confidence
                    cv2.line(
                        frame,
                        (int(person[pt1, 0]), int(person[pt1, 1])),
                        (int(person[pt2, 0]), int(person[pt2, 1])),
                        (255, 0, 0),
                        10,
                    )
        except Exception:
            continue

if __name__ == '__main__':
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)

        if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()  # Convert to numpy array
            draw_skeleton(frame, keypoints)  # Draw skeleton on the frame

        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()