import mediapipe as mp
import math
import cv2
import numpy as np
from config import *

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

def is_hand_raised(landmarks, hand='RIGHT'):
    if landmarks is None: return False
    try:
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST if hand == 'LEFT' else mp_pose.PoseLandmark.RIGHT_WRIST]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER if hand == 'LEFT' else mp_pose.PoseLandmark.RIGHT_SHOULDER]
        return wrist.visibility > 0.7 and shoulder.visibility > 0.7 and wrist.y < shoulder.y
    except (IndexError, TypeError): return False

def is_both_hands_up(landmarks): return is_hand_raised(landmarks, 'RIGHT') and is_hand_raised(landmarks, 'LEFT')

def is_hand_down_base(landmarks, hand='RIGHT'):
    if landmarks is None: return False
    try:
        vis_thresh = 0.6
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST if hand == 'LEFT' else mp_pose.PoseLandmark.RIGHT_WRIST]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW if hand == 'LEFT' else mp_pose.PoseLandmark.RIGHT_ELBOW]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER if hand == 'LEFT' else mp_pose.PoseLandmark.RIGHT_SHOULDER]
        x_cond = wrist.x > shoulder.x if hand == 'LEFT' else wrist.x < shoulder.x
        return wrist.visibility > vis_thresh and elbow.visibility > vis_thresh and shoulder.visibility > vis_thresh and wrist.y > elbow.y and x_cond
    except (IndexError, TypeError): return False

def is_hands_on_hips(landmarks):
    if landmarks is None: return False
    try:
        vis_thresh = 0.7
        l_wrist, r_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_hip, r_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        l_elbow, r_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        if not all(lm.visibility > vis_thresh for lm in [l_wrist,r_wrist,l_hip,r_hip,l_elbow,r_elbow]): return False
        if not (abs(l_wrist.y - l_hip.y) < 0.15 and abs(r_wrist.y - r_hip.y) < 0.15): return False
        if not (l_elbow.x > l_wrist.x and r_elbow.x < r_wrist.x): return False
        return True
    except (IndexError, TypeError): return False

def is_both_hands_down(landmarks): return (is_hand_down_base(landmarks, 'RIGHT') and is_hand_down_base(landmarks, 'LEFT') and not is_hands_on_hips(landmarks))

def is_t_pose(landmarks):
    if landmarks is None: return False
    try:
        l_shoulder, r_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_elbow, r_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        l_wrist, r_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        if not all(lm.visibility > 0.7 for lm in [l_shoulder,r_shoulder,l_elbow,r_elbow,l_wrist,r_wrist]): return False
        if not (abs(l_wrist.y-l_shoulder.y)<0.1 and abs(r_wrist.y-r_shoulder.y)<0.1): return False
        if not (r_wrist.x < r_elbow.x < r_shoulder.x and l_wrist.x > l_elbow.x > l_shoulder.x): return False
        return True
    except (IndexError, TypeError): return False

def is_hands_folded(landmarks):
    if landmarks is None: return False
    try:
        vis_thresh = 0.7
        l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        if not all(lm.visibility > vis_thresh for lm in [l_wrist, r_wrist, l_shoulder, r_shoulder]):
            return False

        wrist_distance = math.sqrt((l_wrist.x - r_wrist.x)**2 + (l_wrist.y - r_wrist.y)**2)
        wrists_below_shoulders = l_wrist.y > l_shoulder.y and r_wrist.y > r_shoulder.y
        shoulder_width = math.sqrt((l_shoulder.x - r_shoulder.x)**2 + (l_shoulder.y - r_shoulder.y)**2)
        
        return wrist_distance < (shoulder_width / 2.5) and wrists_below_shoulders
    except (IndexError, TypeError):
        return False

def get_bounding_box(landmarks, image_shape):
    h, w = image_shape
    x_coords = [int(l.x * w) for l in landmarks if l.visibility > 0.5]
    y_coords = [int(l.y * h) for l in landmarks if l.visibility > 0.5]
    if not x_coords or not y_coords: return None
    return (max(min(x_coords), 0), max(min(y_coords), 0), min(max(x_coords), w), min(max(y_coords), h))

def is_full_body_visible(landmarks):
    required = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]
    return all(landmarks[j].visibility > 0.5 for j in required)

def match_points_to_keypoints(targets, landmarks, image_shape):
    h, w = image_shape
    matched = []
    if landmarks is None: return [False] * len(targets)
    keypoints_to_check = [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    for tx, ty in targets:
        match_found = False
        for lm_idx in keypoints_to_check:
            lm = landmarks[lm_idx]
            if lm.visibility > 0.5:
                lx, ly = int(lm.x * w), int(lm.y * h)
                if np.linalg.norm(np.array([lx, ly]) - np.array([tx, ty])) < MATCH_THRESHOLD:
                    match_found = True
                    break
        matched.append(match_found)
    return matched
