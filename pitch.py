import cv2
import numpy as np

pitch_map = { ## width, height
    1: (.028, .159),
    2: (.124, .159),
    3: (.8718, .159),
    4: (.972, .159),
    5: (.028, .228),
    6: (.0729, .228),
    7: (.124, .228),
    8: (.8718, .228),
    9: (.9231, .228),
    10: (.972, .228),
    11: (.028, .753),
    12: (.0729, .753),
    13: (.124, .753),
    14: (.8718, .753),
    15: (.9231, .753),
    16: (.972, .753),
    17: (.028, .845),
    18: (.124, .845),
    19: (.8718, .845),
    20: (.972, .845),
    21: (.0729, .5),
    22: (.9231, .5),
    23: (.0729, .159),
    24: (.9231, .159),
    25: (.0729, .845),
    26: (.9231, .845),
    27: (.0729, .485),
    28: (.0729, .515),
    29: (.9231, .485),
    30: (.9231, .515),
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

def get_warped_pitch(image_path, frame_pitch_map, pitch='normal'):
    # Load images
    pitch = cv2.imread('pitch.png' if pitch == 'normal' else 'pitch_cb.png')
    pitch = cv2.cvtColor(pitch, cv2.COLOR_RGB2BGR)
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Extract height and width of both images
    pitch_height, pitch_width = pitch.shape[:2]
    frame_height, frame_width = frame.shape[:2]

    # Define corresponding points from both images
    pitch_points = []
    frame_points = []

    for k in frame_pitch_map.keys():
        if k in pitch_map.keys():
            pitch_points.append((int(pitch_width * pitch_map[k][0]), int(pitch_height * pitch_map[k][1])))
            frame_points.append((frame_pitch_map[k][0], frame_pitch_map[k][1]))

    pitch_points = np.array(pitch_points, dtype=np.float32)
    frame_points = np.array(frame_points, dtype=np.float32)

    # Compute homography
    homography_matrix, _ = cv2.findHomography(pitch_points, frame_points, method=cv2.RANSAC)

    # Warp pitch image to match the frame
    return cv2.warpPerspective(pitch, homography_matrix, (frame_width, frame_height))