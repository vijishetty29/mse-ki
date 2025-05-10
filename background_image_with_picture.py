import cv2
import mediapipe as mp
import numpy as np

# Load your custom background image (resize to match webcam resolution)
background_img = cv2.imread("4k-beach-nature-view-s0jgy1y9poz5zg54.jpg")
background_img = cv2.resize(background_img, (640, 480))

# Initialize MediaPipe Pose and Selfie Segmentation
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
pose = mp_pose.Pose()
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    # Get segmentation mask
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seg_results = segmentor.process(rgb_frame)
    mask = seg_results.segmentation_mask

    # Create condition mask
    condition = np.stack((mask,) * 3, axis=-1) > 0.1

    # Composite the person over the background
    foreground = np.where(condition, frame, background_img)

    # Detect and draw pose on the foreground image
    pose_results = pose.process(rgb_frame)
    if pose_results.pose_landmarks:
        draw.draw_landmarks(
            foreground,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        )

    # Show the final output
    cv2.imshow("Virtual Yoga Trainer", foreground)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
