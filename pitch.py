import cv2
import numpy as np

pitch_map = { ## width, height
    1: (.0255, .146),
    2: (.124, .146),
    3: (.8719, .146),
    4: (.9745, .146),
    5: (.0255, .228),
    6: (.0729, .228),
    7: (.124, .228),
    8: (.8719, .228),
    9: (.9231, .228),
    10: (.9745, .228),
    11: (.0255, .753),
    12: (.0729, .753),
    13: (.124, .753),
    14: (.8719, .753),
    15: (.9231, .753),
    16: (.9745, .753),
    17: (.0255, .854),
    18: (.124, .854),
    19: (.8719, .854),
    20: (.9745, .854),
}

frame_pitch_map = { ## width mul, height mul
    1: (.529, .973),
    2: (.366, .742),
    3: (.183, .497),
    5: (.62, .95),
    6: (.498, .815),
    7: (.422, .733),
    8: (.198, .497),
    12: (.89, .742),
    13: (.762, .687),
}

def get_warped_pitch():
    # Load images
    pitch = cv2.imread('pitch.png')
    pitch = cv2.cvtColor(pitch, cv2.COLOR_RGB2BGR)
    frame = cv2.imread('frame.png')
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Extract height and width of both images
    pitch_height, pitch_width = pitch.shape[:2]
    frame_height, frame_width = frame.shape[:2]

    # Define corresponding points from both images
    pitch_points = []
    frame_points = []

    for k in frame_pitch_map.keys():
        pitch_points.append((int(pitch_width * pitch_map[k][0]), int(pitch_height * pitch_map[k][1])))
        frame_points.append((int(frame_width * frame_pitch_map[k][0]), int(frame_height * frame_pitch_map[k][1])))

    pitch_points = np.array(pitch_points, dtype=np.float32)
    frame_points = np.array(frame_points, dtype=np.float32)

    # Compute homography
    homography_matrix, _ = cv2.findHomography(pitch_points, frame_points, method=cv2.RANSAC)

    # Warp pitch image to match the frame
    warped_pitch = cv2.warpPerspective(pitch, homography_matrix, (frame_width, frame_height))
    return cv2.cvtColor(warped_pitch, cv2.COLOR_BGR2RGB)
