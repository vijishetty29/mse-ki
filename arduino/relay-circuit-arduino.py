import cv2
import mediapipe as mp
import serial
import time

# --- Serial Communication Setup ---
SERIAL_PORT = '/dev/cu.usbserial-2130'
BAUD_RATE = 9600

ser = None # Initialize ser to None
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to Arduino on {SERIAL_PORT} at {BAUD_RATE} baud.")
except serial.SerialException as e:
    print(f"Error: Could not open serial port {SERIAL_PORT}. {e}")
    print("Please check if the Arduino is connected and the port is correct.")
    print("You might need to close the Arduino IDE if it's open.")
    exit()

# --- MediaPipe Pose Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- OpenCV Camera Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    if ser and ser.is_open: # Ensure ser is open before trying to close
        ser.close()
    exit()

# --- Pose Logic Variables ---
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
WRIST_UP_THRESHOLD = -0.10
WRIST_DOWN_THRESHOLD = 0.10

last_command_sent = None

def send_command_to_arduino(command):
    global last_command_sent
    if command != last_command_sent:
        try:
            ser.write(command.encode())
            print(f"Sent command: {command}")
            last_command_sent = command
        except Exception as e:
            print(f"Error sending data: {e}")

print("\n--- Starting Pose Detection ---")
print("Move your left hand: Up to turn Relay 1 ON, Down to turn Relay 2 ON, Neutral for ALL OFF.")

# --- Main Program Logic within try...finally ---
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                
                left_shoulder_y = landmarks[LEFT_SHOULDER].y
                left_wrist_y = landmarks[LEFT_WRIST].y

                relative_wrist_y = left_wrist_y - left_shoulder_y
                
                # --- Pose Logic to Send Serial Commands ---
                if relative_wrist_y < WRIST_UP_THRESHOLD:
                    send_command_to_arduino('1') # Command for Relay 1 ON
                    cv2.putText(frame, "RELAY 1 ON", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif relative_wrist_y > WRIST_DOWN_THRESHOLD:
                    send_command_to_arduino('2') # Command for Relay 2 ON
                    cv2.putText(frame, "RELAY 2 ON", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    send_command_to_arduino('0') # Command for ALL OFF
                    cv2.putText(frame, "ALL RELAYS OFF", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            except Exception as e:
                pass # Suppress frequent errors if not all landmarks are visible

        cv2.imshow('MediaPipe Pose & Arduino Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- Cleanup actions that always run ---
    print("\n--- Cleaning up... ---")
    if ser and ser.is_open:
        # Send '0' command to turn off all relays before closing the serial port
        try:
            ser.write(b'0') # Send '0' as bytes
            print("Sent '0' command to Arduino (ALL OFF).")
            time.sleep(0.1) # Give Arduino a moment to process
        except Exception as e:
            print(f"Error sending final OFF command: {e}")
        ser.close() # Close the serial port
        print("Serial port closed.")
    else:
        print("Serial port was not open or already closed.")

    cap.release()
    cv2.destroyAllWindows()
    print("Program ended.")