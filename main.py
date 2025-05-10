import cv2
import numpy as np
import mediapipe as mp
import random
import pygame
import time
import math

# Initialize pygame for sound
pygame.mixer.init()
match_sound = pygame.mixer.Sound("sounds/ding.mp3")  # match sound
unmatch_sound = pygame.mixer.Sound("sounds/unmatch.mp3")  # unmatch sound
celebrate_sound = pygame.mixer.Sound("sounds/celebrate.mp3")  # unmatch sound

# Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Config
TARGET_RADIUS = 20
MATCH_THRESHOLD = 40
TARGETS_PER_CYCLE = 6
TARGET_CYCLES = 1
CYCLE_TIME_LIMIT = 60  # seconds

# Cycle order option: "random" or "predefined"
CYCLE_ORDER = "predefined"  # Set this to "random" for random cycles or "predefined" for fixed cycle order

# Session variables
bounding_box_frozen = False
target_points = []
matched_flags = []
successful_cycles = 0
total_cycles_attempted = 0
session_complete = False
show_confetti = False
confetti_start_time = None
cycle_start_time = None
current_cycle_type = None  # We'll set this dynamically based on the cycle order
predefined_order = ["arrow"]

# Party Popper Confetti
def draw_confetti(frame):
    for _ in range(50):
        x = random.randint(0, frame.shape[1])
        y = random.randint(0, frame.shape[0])
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        cv2.circle(frame, (x, y), random.randint(5, 15), color, -1)

def draw_bow_and_arrow_points(frame, center_x, center_y, size=200):
    shaft_length = 300
    arrow_tip = (center_x - shaft_length - 50 // 2, center_y)
    arrow_start = (center_x + shaft_length // 2, center_y)

    # Bow arc dimensions
    a = shaft_length // 2
    b = int(size * 0.8)
    cx, cy = center_x - 10, center_y

    # Compute start and end points on the ellipse (270° to 90°)
    angle_start = 90
    angle_end = 270
    rad_start = np.deg2rad(angle_start)
    rad_end = np.deg2rad(angle_end)

    bow_top = (
        int(cx + a * np.cos(rad_start)),
        int(cy + b * np.sin(rad_start))
    )
    bow_bottom = (
        int(cx + a * np.cos(rad_end)),
        int(cy + b * np.sin(rad_end))
    )

    # Draw arrow shaft
    cv2.line(frame, arrow_start, arrow_tip, (255, 255, 255), 4)

    # Draw arrowhead
    head_size = 20
    arrowhead_top = (arrow_tip[0] - head_size, arrow_tip[1] - head_size)
    arrowhead_bottom = (arrow_tip[0] - head_size, arrow_tip[1] + head_size)
    # cv2.line(frame, arrow_tip, arrowhead_top, (255, 255, 255), 4)
    # cv2.line(frame, arrow_tip, arrowhead_bottom, (255, 255, 255), 4)

    # Draw bow arc
    cv2.ellipse(frame, (cx, cy), (a, b), 0, angle_start, angle_end, (255, 255, 255), 4)
    cv2.line(frame, bow_top, bow_bottom, (255, 255, 255), 4)

    # Draw the four target points
    target_points = [bow_top, bow_bottom, arrow_start, arrow_tip]


    return target_points


# Utility functions
def get_bounding_box(landmarks, image_shape):
    h, w = image_shape
    x_coords = [int(l.x * w) for l in landmarks if l.visibility > 0.5]
    y_coords = [int(l.y * h) for l in landmarks if l.visibility > 0.5]
    if not x_coords or not y_coords:
        return None
    return (max(min(x_coords), 0), max(min(y_coords), 0),
            min(max(x_coords), w), min(max(y_coords), h))

def is_full_body_visible(landmarks):
    required = [mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE]
    return all(landmarks[j].visibility > 0.5 for j in required)

def is_hand_above_head_with_ankle_visible(landmarks):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Wrist above head (nose y)
    wrist_above_head = (
        (left_wrist.visibility > 0.5 and nose.visibility > 0.5 and left_wrist.y < nose.y) or
        (right_wrist.visibility > 0.5 and nose.visibility > 0.5 and right_wrist.y < nose.y)
    )

    # At least one ankle visible
    ankle_visible = (
        (left_ankle.visibility > 0.5) or (right_ankle.visibility > 0.5)
    )

    return wrist_above_head and ankle_visible


def generate_target_points(bbox, num_targets, pattern="random"):
    x_min, y_min, x_max, y_max = bbox
    margin = 20

    if pattern == "chair":
        # Chair pattern: 2 points for "seat" and 2 for "backrest"
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        targets = [
            (center_x - 50, center_y + 100),  # Seat left
            (center_x + 50, center_y + 100),  # Seat right
            (center_x - 50, center_y - 100),  # Backrest left
            (center_x + 50, center_y - 100)   # Backrest right
        ]
        return targets

    elif pattern == "arrow":
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        return draw_bow_and_arrow_points(bbox, center_x, center_y, 200)
        
    elif pattern == "star":
        # Star pattern: Points arranged in a star shape
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        radius = 200
        angle_step = 72  # 360° / 5 points (5-pointed star)
        targets = []
        
        # Calculate the 5 points of the star
        for i in range(5):
            angle = math.radians(i * angle_step)
            target_x = int(center_x + radius * math.cos(angle))
            target_y = int(center_y + radius * math.sin(angle))
            targets.append((target_x, target_y))

        # Draw the lines between the points to form the star
        for i in range(5):
            # Connect each point to the next, forming a star
            start_point = targets[i]
            end_point = targets[(i + 2) % 5]  # Connect each point to the second next
            cv2.line(frame, start_point, end_point, (255, 255, 255), 4)  # White lines

          # Red points
        return targets

    else:
        # Random pattern: Random targets within bounding box
        return [(random.randint(x_min - margin, x_max + margin),
                 random.randint(y_min - margin, y_max + margin)) for _ in range(num_targets)]

def match_points_to_keypoints(targets, landmarks, image_shape):
    h, w = image_shape
    matched = []
    for tx, ty in targets:
        match_found = False
        for lm in landmarks:
            if lm.visibility < 0.5:
                continue
            lx, ly = int(lm.x * w), int(lm.y * h)
            dist = np.linalg.norm(np.array([lx, ly]) - np.array([tx, ty]))
            if dist < MATCH_THRESHOLD:
                match_found = True
                break
        matched.append(match_found)
    return matched

# Start camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks and not session_complete:
        landmarks = results.pose_landmarks.landmark

        # Setup for first time
        if False or (not bounding_box_frozen and is_full_body_visible(landmarks)):
            bbox = get_bounding_box(landmarks, frame.shape[:2])
            if bbox:
                bounding_box_frozen = True
                frozen_box = bbox
                
                # Initialize first cycle based on the selected order
                if CYCLE_ORDER == "random":
                    current_cycle_type = "random"  # Generate random targets each cycle
                else:
                    # Predefined order, cycling through patterns
                    current_cycle_type = predefined_order[total_cycles_attempted % len(predefined_order)]
                
                target_points = generate_target_points(frozen_box, TARGETS_PER_CYCLE, current_cycle_type)
                cycle_start_time = time.time()

        if bounding_box_frozen:
            x1, y1, x2, y2 = frozen_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            matched_flags = match_points_to_keypoints(target_points, landmarks, frame.shape[:2])
            all_matched = all(matched_flags)

            # Draw targets
            for i, (x, y) in enumerate(target_points):
                color = (0, 255, 0) if matched_flags[i] else (0, 0, 255)
                cv2.circle(frame, (x, y), TARGET_RADIUS, color, -1)

            # Timer handling
            elapsed = time.time() - cycle_start_time
            time_left = max(0, int(CYCLE_TIME_LIMIT - elapsed))
            cv2.putText(frame, f"Time Left: {time_left}s", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Cycle: {total_cycles_attempted+1}/{TARGET_CYCLES}", (30, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # After elapsed time or match complete:
            if elapsed > CYCLE_TIME_LIMIT or all_matched:

                # Always increment total attempted cycles
                total_cycles_attempted += 1

                if all_matched and elapsed <= CYCLE_TIME_LIMIT:
                    successful_cycles += 1
                    match_sound.play()
                else:
                    unmatch_sound.play()
                    pass

                # Check session complete
                if total_cycles_attempted >= TARGET_CYCLES:
                    session_complete = True
                else:
                    # Prepare next cycle based on the selected order
                    if CYCLE_ORDER == "random":
                        target_points = generate_target_points(frozen_box, TARGETS_PER_CYCLE, "random")
                    else:
                        current_cycle_type = predefined_order[total_cycles_attempted % len(predefined_order)]
                        target_points = generate_target_points(frozen_box, TARGETS_PER_CYCLE, current_cycle_type)

                    cycle_start_time = time.time()

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    elif session_complete:
        cv2.putText(frame, f"Session Complete! Score: {successful_cycles}/{TARGET_CYCLES}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(frame, "Press 'N' to start a new session or ESC to exit.",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        if successful_cycles == TARGET_CYCLES and not show_confetti:
            show_confetti = True
            confetti_start_time = time.time()

        if show_confetti:
            draw_confetti(frame)
            celebrate_sound.play()
            if time.time() - confetti_start_time > 3:  # Show confetti for 3 seconds
                show_confetti = False
                successful_cycles = 0
                session_complete = False
                bounding_box_frozen=False

    # Show frame
    cv2.imshow("Pose Matching Game", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


