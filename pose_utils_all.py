import cv2
import numpy as np
import mediapipe as mp
import random
import pygame
import time
import math
import json
from collections import deque
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image, ImageDraw, ImageFont

# --- Serial Port Integration ---
class MockSerial:
    def __init__(self, port=None, baudrate=None, timeout=None):
        self._port = port
        self.is_open = True # Added for compatibility with cleanup logic
        print(f"MOCK SERIAL: Initialized.")
    def write(self, data):
        print(f"MOCK SERIAL >> Sent: {data.decode().strip()}"); return len(data)
    def close(self):
        self.is_open = False
        print(f"MOCK SERIAL: Port closed.")

try:
    import serial
except ImportError:
    print("WARNING: pyserial library not found."); serial = type('module', (object,), {'Serial': MockSerial})

# --- Pygame Sound Initialization ---
pygame.mixer.init()
try:
    match_sound = pygame.mixer.Sound("assets/sounds/ding.mp3")
    unmatch_sound = pygame.mixer.Sound("assets/sounds/unmatch.mp3")
    celebrate_sound = pygame.mixer.Sound("assets/sounds/celebrate.mp3")
    combo_break_sound = pygame.mixer.Sound("assets/sounds/unmatch.mp3")
    hit_sound = pygame.mixer.Sound("assets/sounds/ding.mp3")
    slice_sound = pygame.mixer.Sound("assets/sounds/slice.mp3") # New sound for slicing
    bomb_sound = pygame.mixer.Sound("assets/sounds/bomb.mp3") # New sound for bomb explosion
except pygame.error as e:
    print(f"Error loading sound files: {e}. Using dummy sounds.")
    match_sound,unmatch_sound,celebrate_sound,combo_break_sound,hit_sound, slice_sound, bomb_sound = [pygame.mixer.Sound(np.array([0],dtype=np.int16)) for _ in range(7)]

# --- MediaPipe Pose Initialization ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Game Configuration ---
POSE_HOLD_TIME = 2.0; SESSION_COOLDOWN = 3; FEEDBACK_DISPLAY_DURATION = 2.0
TARGET_RADIUS = 20; MATCH_THRESHOLD = 40; TARGETS_PER_CYCLE = 4; TARGET_CYCLES = 3; CYCLE_TIME_LIMIT = 30
YOGA_POSE_IMAGES = ['assets/yoga_poses/goddess.jpg',
    'assets/yoga_poses/your_reference_pose.jpg', # Replace with a real image path or remove
    'assets/yoga_poses/high-lunge.jpg',
    'assets/yoga_poses/post-big-image.jpg','assets/yoga_poses/warrior.jpg',
    'assets/yoga_poses/side-bend.jpg',]; YOGA_SIMILARITY_THRESHOLD = 0.95; YOGA_CYCLE_TIME_LIMIT = 30; YOGA_POSE_HOLD_DURATION = 1.0;
RHYTHM_SEQUENCE_LENGTH = 10; RHYTHM_TARGET_TIME_LIMIT = 3.0
DODGER_INITIAL_LIVES = 3; DODGER_SPAWN_RATE = 0.5;
DODGER_POINTS_PER_ITEM = 50
DODGER_MAGNET_RANGE = 200
DODGER_MAGNET_STRENGTH = 0.1
NINJA_SLICER_INITIAL_LIVES = 3
NINJA_SLICER_SPAWN_RATE = 0.8 # Slightly slower spawn rate
NINJA_SLICER_FRUIT_CHANCE = 0.8
NINJA_SLICER_POINTS_PER_FRUIT = 10
TRAIL_LENGTH = 10 # Longer trail for smoother slicing


# --- Pattern and Sequence Generation ---
def generate_random_pattern(bbox, num_targets):
    x_min, y_min, x_max, y_max = bbox; margin = 20
    return [(random.randint(x_min - margin, x_max + margin), random.randint(y_min - margin, y_max + margin)) for _ in range(num_targets)]

def generate_target_points(bbox, num_targets, pattern="random"):
    return generate_random_pattern(bbox, num_targets)


# --- Pose Detection Functions ---
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
        vis_thresh = 0.6; wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST if hand == 'LEFT' else mp_pose.PoseLandmark.RIGHT_WRIST]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW if hand == 'LEFT' else mp_pose.PoseLandmark.RIGHT_ELBOW]; shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER if hand == 'LEFT' else mp_pose.PoseLandmark.RIGHT_SHOULDER]
        x_cond = wrist.x > shoulder.x if hand == 'LEFT' else wrist.x < shoulder.x
        return wrist.visibility>vis_thresh and elbow.visibility>vis_thresh and shoulder.visibility>vis_thresh and wrist.y>elbow.y and x_cond
    except (IndexError, TypeError): return False

def is_hands_on_hips(landmarks):
    if landmarks is None: return False
    try:
        vis_thresh = 0.7; l_wrist, r_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_hip, r_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_HIP]; l_elbow, r_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        if not all(lm.visibility > vis_thresh for lm in [l_wrist,r_wrist,l_hip,r_hip,l_elbow,r_elbow]): return False
        if not (abs(l_wrist.y - l_hip.y) < 0.15 and abs(r_wrist.y - r_hip.y) < 0.15): return False
        if not (l_elbow.x > l_wrist.x and r_elbow.x < r_wrist.x): return False
        return True
    except (IndexError, TypeError): return False

def is_both_hands_down(landmarks): return (is_hand_down_base(landmarks, 'RIGHT') and is_hand_down_base(landmarks, 'LEFT') and not is_hands_on_hips(landmarks))

def is_t_pose(landmarks):
    if landmarks is None: return False
    try:
        l_shoulder, r_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]; l_elbow, r_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
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

# --- Utility and Drawing Functions ---
def get_bounding_box(landmarks, image_shape):
    h, w = image_shape; x_coords=[int(l.x*w) for l in landmarks if l.visibility>0.5]; y_coords=[int(l.y*h) for l in landmarks if l.visibility>0.5]
    if not x_coords or not y_coords: return None
    return (max(min(x_coords),0), max(min(y_coords),0), min(max(x_coords),w), min(max(y_coords),h))

def is_full_body_visible(landmarks):
    required = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE]
    return all(landmarks[j].visibility > 0.5 for j in required)

def match_points_to_keypoints(targets, landmarks, image_shape):
    h, w = image_shape; matched = []
    if landmarks is None: return [False]*len(targets)
    keypoints_to_check = [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    for tx, ty in targets:
        match_found = False;
        for lm_idx in keypoints_to_check:
            lm = landmarks[lm_idx]
            if lm.visibility > 0.5:
                lx, ly = int(lm.x * w), int(lm.y * h)
                if np.linalg.norm(np.array([lx,ly])-np.array([tx,ty])) < MATCH_THRESHOLD: match_found=True; break
        matched.append(match_found)
    return matched

def extract_landmarks_from_image(image_path, pose_estimator):
    image = cv2.imread(image_path)
    if image is None: return None, None
    results = pose_estimator.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        h, w, _ = image.shape
        return np.array([[lmk.x * w, lmk.y * h] for lmk in results.pose_landmarks.landmark]), image
    return None, image

def calculate_similarity(landmarks1, landmarks2):
    if landmarks1.shape != landmarks2.shape: return 0.0
    l1_norm = landmarks1 - np.mean(landmarks1, axis=0); l2_norm = landmarks2 - np.mean(landmarks2, axis=0)
    norm1 = np.linalg.norm(l1_norm.flatten()); norm2 = np.linalg.norm(l2_norm.flatten())
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(l1_norm.flatten(), l2_norm.flatten()) / (norm1 * norm2)

def load_reference_poses_yoga():
    yoga_landmarks_list, yoga_images = [], []
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose_static:
        for img_path in YOGA_POSE_IMAGES:
            landmarks, img = extract_landmarks_from_image(img_path, pose_static)
            if landmarks is not None: yoga_landmarks_list.append(landmarks); yoga_images.append(cv2.resize(img, (200, 200)))
    return yoga_landmarks_list, yoga_images

class PoseGameWindow(QtWidgets.QMainWindow):
    STATE_MENU, STATE_GAME_MODE_SELECTION, STATE_IN_GAME_MENU = -3, -2, -1
    STATE_STATS = -5
    STATE_LANGUAGE_SELECTION = -4 
    
    STATE_WAITING_FOR_PLAYER, STATE_PLAYING_CYCLE, STATE_CYCLE_END_FEEDBACK, STATE_SESSION_COMPLETE = 0, 1, 2, 3
    STATE_PLAYING_RHYTHM, STATE_PLAYING_DODGER, STATE_PLAYING_NINJA_SLICER = 4, 5, 6

    def __init__(self):
        super().__init__(); self.setWindowTitle("Pose Matching Game"); self.setGeometry(100, 100, 1280, 720); self.showFullScreen()

        self.font_path = None
        try:
            possible_fonts = [
                "C:/Windows/Fonts/arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/liberation/LiberationSans-Regular.ttf"
            ]
            for font in possible_fonts:
                try:
                    ImageFont.truetype(font, 10)
                    self.font_path = font
                    print(f"INFO: Using font: {self.font_path}")
                    break
                except IOError:
                    continue
            if not self.font_path:
                 print("WARNING: Could not find a default font. Special characters may not render correctly.")
        except Exception as e:
            print(f"ERROR: Font loading failed: {e}")

        # --- MODIFIED: Load All Game Assets ---
        print("--- Loading Game Assets ---")
        self.watermelon_img = self.load_image_asset("assets/images/watermelon.png", (80, 80))
        self.watermelon_half_top_img = self.load_image_asset("assets/images/watermelon_half_top.png", (80, 45))
        self.watermelon_half_bottom_img = self.load_image_asset("assets/images/watermelon_half_bottom.png", (80, 45))
        self.bomb_img = self.load_image_asset("assets/images/bomb.png", (70, 70))
        self.splash_img = self.load_image_asset("assets/images/splash.png", (100, 100))
        self.magnet_img_assets = self.load_image_asset("assets/images/magnet.png", (60, 60))
        self.gear_img_assets = self.load_image_asset("assets/images/gear.png", (50, 50))
        self.nuts_img_assets = self.load_image_asset("assets/images/nuts.png", (50, 50))
        self.apple_img_assets = self.load_image_asset("assets/images/apple.png", (50, 50))
        print("--------------------------")
        
        self.leaderboard = []
        self.stats = {}
        self.load_leaderboard()
        self.load_stats()
        self.session_nuts_collected = 0
        self.session_apples_hit = 0

        self.central_widget = QtWidgets.QWidget(); self.setCentralWidget(self.central_widget); self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.video_label = QtWidgets.QLabel("Waiting..."); self.video_label.setAlignment(QtCore.Qt.AlignCenter); self.layout.addWidget(self.video_label)
        self.game_state = self.STATE_MENU; self.current_game_mode = None; self.previous_game_state = None; self.bounding_box_frozen = False; self.frozen_box = None; self.last_session_end_time = None
        self.score = 0; self.combo = 1; self.feedback_text = ""; self.feedback_text_alpha = 0; self.feedback_display_start_time = 0
        self.mode_selection_entry_time = None 
        self.target_points, self.matched_flags, self.successful_cycles, self.total_cycles_attempted, self.cycle_start_time, self.cycle_end_message = [], [], 0, 0, None, ""
        self.rhythm_sequence, self.rhythm_current_step, self.rhythm_target_start_time = [], 0, 0
        self.dodger_objects, self.dodger_lives, self.dodger_start_time, self.dodger_spawn_timer = [], 0, 0, 0
        self.dodger_current_spawn_rate = DODGER_SPAWN_RATE
        self.dodger_last_difficulty_increase = 0
        self.ninja_slicer_objects, self.ninja_slicer_lives, self.ninja_slicer_spawn_timer = [], 0, 0
        self.left_hand_trail, self.right_hand_trail = deque(maxlen=TRAIL_LENGTH), deque(maxlen=TRAIL_LENGTH)
        self.left_hand_rect, self.right_hand_rect = None, None
        
        self.session_snapshots = []; self.display_snapshots = []
        self.pause_start_time = None
        self.relay_on_duration = 30; self.left_relay_on, self.right_relay_on = False, False; self.left_relay_timer_start, self.right_relay_timer_start, self.serial_port = None, None, None
        
        # --- Confetti Initialization ---
        self.confetti_particles = []
        self.confetti_triggered_this_session = False
        
        self.last_command_sent = None
        try:
            port_name = '/dev/cu.usbserial-2130'
            self.serial_port = serial.Serial(port_name, 9600, timeout=1)
            print(f"INFO: Opened serial port {port_name}.")
        except Exception as e:
            print(f"WARNING: Serial port error: {e}. Using mock port.")
            self.serial_port = MockSerial()
        
        self.yoga_landmarks_list, self.yoga_images = load_reference_poses_yoga(); self.current_yoga_pose_index = 0; self.yoga_pose_match_start_time = None; self.yoga_highest_similarity_score = 0.0
        self.pose_start_time = {k: None for k in ["start_game", "random_mode", "yoga_mode", "rhythm_mode", "dodger_mode", "ninja_slicer_mode", "back_to_main_from_mode_select", "resume_game", "ingame_main_menu", "ingame_exit", "t_pose_menu", "restart_session", "relay_left_on", "relay_right_on", "no_pose_detected_pause", "view_stats", "select_language", "language_en", "language_de", "relay_left_on_end_screen", "relay_right_on_end_screen"]}
        
        self.translations = {
            'en': {
                'pose_to_play': "Pose To Play", 'start_game': "Start Game (Both Hands Up)", 'test_relays': "Use Right Hand for Fan & Water", 'test_relays_end': "Use Left Hand for Fan & Right Hand for Water",
                'select_game_mode': "Select Game Mode", 'random_mode': "Random (L Hand)", 'yoga_mode': "Yoga (R Hand)",
                'rhythm_mode': "Rhythm (T-Pose)", 'collector_mode': "Collector (Hips)", 'ninja_slicer_mode': "Ninja Slicer (Hands Folded)", 'stats_mode': "Stats (Hands Down)",
                'player_stats': "PLAYER STATS", 'high_score': "High Score", 'games_played': "Games Played",
                'total_nuts_collected': "Total Nuts Collected", 'back_to_menu': "Pose with hands down to go back",
                'stand_center': "Stand in the center of the frame!", 'game_paused': "Game Paused", 'resume': "Resume (Right Hand Up)",
                'main_menu': "Go to Main Menu (Left Hand Up)", 'success': "SUCCESS!", 'missed': "MISSED!", 'pose_success': "POSE SUCCESS!",
                'pose_missed': "POSE MISSED!", 'hit': "HIT!", 'ouch': "OUCH!", 'game_over': "Game Over!", 'session_complete': "Session Complete!",
                'final_score': "Final Score", 'nuts_collected': "Nuts Collected", 'poses_completed': "Poses Completed",
                'both_hands_up_menu': "Both Hands Up for Menu", 'score': "Score", 'lives': "Lives", 'cycles': "Cycles", 'combo': "Combo",
                'time': "Time", 'left_relay': "Fan", 'right_relay': "Water", 'left_relay_on_status': "Fan: ON",
                'left_relay_off_status': "Fan: OFF", 'right_relay_on_status': "Water: ON", 'right_relay_off_status': "Water: OFF",
                'select_language_prompt': "Use Left Hand to select language", 'english': "English", 'german': "Deutsch",
                'mode_Collector': "Collector", 'mode_Random': "Random", 'mode_Rhythm': "Rhythm", 'mode_Yoga': "Yoga", 'mode_NinjaSlicer': "Ninja Slicer"
            },
            'de': {
                'pose_to_play': "Posieren zum Spielen", 'start_game': "Spiel starten (Beide Hände hoch)", 'test_relays': "Rechte Hand für Lüfter & Wasser",'test_relays_end': "Linke Hand für Lüfter & Rechte Hand für Wasser",
                'select_game_mode': "Spielmodus auswählen", 'random_mode': "Zufall (Linke Hand)", 'yoga_mode': "Yoga (Rechte Hand)",
                'rhythm_mode': "Rhythmus (T-Pose)", 'collector_mode': "Sammler (Hüften)", 'ninja_slicer_mode': "Ninja Slicer (Hände gefaltet)", 'stats_mode': "Statistiken (Hände runter)",
                'player_stats': "SPIELERSTATISTIKEN", 'high_score': "Highscore", 'games_played': "Gespielte Spiele",
                'total_nuts_collected': "Gesammelte Nüsse", 'back_to_menu': "Hände senken, um zurück zu gehen",
                'stand_center': "Stellen Sie sich in die Mitte des Bildes!", 'game_paused': "Spiel pausiert",
                'resume': "Fortsetzen (Rechte Hand hoch)", 'main_menu': "Zum Hauptmenü (Linke Hand hoch)", 'success': "ERFOLG!",
                'missed': "VERPASST!", 'pose_success': "POSE ERFOLG!", 'pose_missed': "POSE VERPASST!", 'hit': "TREFFER!", 'ouch': "AUTSCH!",
                'game_over': "Spiel vorbei!", 'session_complete': "Sitzung beendet!", 'final_score': "Endstand",
                'nuts_collected': "Nüsse gesammelt", 'poses_completed': "Posen abgeschlossen",
                'both_hands_up_menu': "Beide Hände hoch für Menü", 'score': "Punktzahl", 'lives': "Leben", 'zyklen': "Zyklen",
                'combo': "Kombo", 'time': "Zeit", 'left_relay': "Lüfter", 'right_relay': "Wasser",
                'left_relay_on_status': "Lüfter: AN", 'left_relay_off_status': "Lüfter: AUS",
                'right_relay_on_status': "Wasser: AN", 'right_relay_off_status': "Wasser: AUS",
                'select_language_prompt': "Linke Hand zur Sprachauswahl", 'english': "English", 'german': "Deutsch",
                'mode_Collector': "Sammler", 'mode_Random': "Zufall", 'mode_Rhythm': "Rhythmus", 'mode_Yoga': "Yoga", 'mode_NinjaSlicer': "Ninja Slicer"
            }
        }
        self.current_language = 'de'
        self.current_translation = self.translations[self.current_language]

        self.cap = cv2.VideoCapture(0); self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.update_frame); self.timer.start(30)
        print("Game Initialized.")

    def send_command_to_arduino(self, command_byte):
        """ Sends a command to the Arduino, avoiding redundant sends. """
        if command_byte != self.last_command_sent:
            try:
                self.serial_port.write(command_byte)
                print(f"Sent command to Arduino: {command_byte.decode()}")
                self.last_command_sent = command_byte
            except Exception as e:
                print(f"Error sending data to Arduino: {e}")

    def draw_text(self, img, text, org, font_size, color, align="left", alpha=255):
        if not self.font_path or not text:
            if text:
                try: 
                    cv2.putText(img, text.encode('utf-8').decode('ascii', 'ignore'), org, cv2.FONT_HERSHEY_SIMPLEX, font_size / 35.0, color, 2)
                except:
                    pass
            return

        font = ImageFont.truetype(self.font_path, size=font_size)
        try:
            bbox = font.getbbox(text)
        except AttributeError:
            w, h = font.getsize(text)
            bbox = (0, 0, w, h)

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x, y = org
        y = y - bbox[1] 

        if align == "right":
            x = org[0] - text_width
        elif align == "center":
            x = org[0] - text_width // 2

        text_img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_img)
        
        rgb_color = (color[2], color[1], color[0])
        draw.text((-bbox[0], -bbox[1]), text, font=font, fill=rgb_color)
        
        text_np = np.array(text_img)
        
        frame_h, frame_w, _ = img.shape
        x_start, y_start = int(x), int(y)
        x_end, y_end = int(x + text_width), int(y + text_height)

        if x_start >= frame_w or y_start >= frame_h or x_end <= 0 or y_end <= 0:
            return 

        x_start_f = max(x_start, 0)
        y_start_f = max(y_start, 0)
        x_end_f = min(x_end, frame_w)
        y_end_f = min(y_end, frame_h)

        text_x_start = max(0, -x_start)
        text_y_start = max(0, -y_start)
        text_x_end = text_x_start + (x_end_f - x_start_f)
        text_y_end = text_y_start + (y_end_f - y_start_f)

        if text_x_end > text_x_start and text_y_end > text_y_start:
            roi = img[y_start_f:y_end_f, x_start_f:x_end_f]
            text_slice = text_np[text_y_start:text_y_end, text_x_start:text_x_end]
            
            alpha_values = text_slice[:, :, 3] / 255.0 * (alpha / 255.0)
            alpha_mask = alpha_values[:, :, np.newaxis]
            
            blended_roi = roi * (1 - alpha_mask) + text_slice[:, :, :3] * alpha_mask
            img[y_start_f:y_end_f, x_start_f:x_end_f] = blended_roi.astype(np.uint8)

    def set_language(self, lang_code):
        self.current_language = lang_code
        self.current_translation = self.translations[lang_code]
        print(f"Language set to: {lang_code}")

    def load_leaderboard(self):
        try:
            with open('leaderboard.json', 'r') as f:
                self.leaderboard = json.load(f)
            print("Leaderboard loaded successfully.")
        except (FileNotFoundError, json.JSONDecodeError):
            self.leaderboard = []
            print("No leaderboard file found, starting fresh.")

    def save_leaderboard(self):
        with open('leaderboard.json', 'w') as f:
            json.dump(self.leaderboard, f, indent=4)

    def load_stats(self):
        try:
            with open('stats.json', 'r') as f:
                self.stats = json.load(f)
            print("Stats loaded successfully.")
        except (FileNotFoundError, json.JSONDecodeError):
            self.stats = {
                "high_scores": {"Collector": 0, "Rhythm": 0, "Random": 0, "Yoga": 0, "NinjaSlicer": 0},
                "games_played": {"Collector": 0, "Rhythm": 0, "Random": 0, "Yoga": 0, "NinjaSlicer": 0},
                "total_nuts_collected": 0
            }
            print("No stats file found, starting fresh.")

    def save_stats(self):
        with open('stats.json', 'w') as f:
            json.dump(self.stats, f, indent=4)
            
    def update_leaderboard_and_stats(self):
        mode_map = {'dodger': 'Collector', 'random': 'Random', 'rhythm': 'Rhythm', 'yoga': 'Yoga', 'ninja_slicer': 'NinjaSlicer'}
        mode_name = mode_map.get(self.current_game_mode)
        if not mode_name: return

        self.stats["games_played"][mode_name] = self.stats["games_played"].get(mode_name, 0) + 1
        
        if mode_name == "Collector":
            self.stats["total_nuts_collected"] = self.stats.get("total_nuts_collected", 0) + self.session_nuts_collected
        
        if self.score > self.stats["high_scores"].get(mode_name, 0):
            self.stats["high_scores"][mode_name] = self.score
        
        self.save_stats()

        if mode_name in ["Collector", "Rhythm", "Random", "NinjaSlicer"]:
            if len(self.leaderboard) < 5 or self.score > self.leaderboard[-1].get('score', 0):
                self.leaderboard.append({'mode': mode_name, 'score': self.score})
                self.leaderboard.sort(key=lambda x: x['score'], reverse=True)
                self.leaderboard = self.leaderboard[:5]
                self.save_leaderboard()

    def load_image_asset(self, path, size):
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None: raise FileNotFoundError(f"Asset not found at {path}")
            img = cv2.resize(img, size)
            if img.shape[2] == 4:
                bgr, alpha = img[:, :, 0:3], img[:, :, 3]
            else:
                bgr, alpha = img, np.ones(img.shape[:2], dtype=img.dtype) * 255
            print(f"INFO: Asset '{path}' loaded successfully.")
            return {'bgr': bgr, 'alpha': alpha, 'size': size}
        except Exception as e:
            print(f"WARNING: Could not load asset '{path}'. Reason: {e}")
            # Return a placeholder asset
            placeholder_bgr = np.zeros((size[1], size[0], 3), np.uint8)
            cv2.putText(placeholder_bgr, "?", (size[0]//4, size[1]*3//4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            placeholder_alpha = np.ones((size[1], size[0]), dtype=np.uint8) * 255
            return {'bgr': placeholder_bgr, 'alpha': placeholder_alpha, 'size': size}


    def start_game_mode(self, mode):
        self.reset_game_progress(); self.current_game_mode = mode
        if mode in ['random', 'yoga', 'rhythm', 'dodger', 'ninja_slicer']: self.game_state = self.STATE_WAITING_FOR_PLAYER; print(f"Starting {mode.title()} Mode...")

    def reset_game_progress(self):
        self.score, self.combo, self.feedback_text_alpha = 0, 1, 0; self.bounding_box_frozen, self.frozen_box, self.last_session_end_time = False, None, None
        self.successful_cycles, self.total_cycles_attempted, self.session_snapshots = 0, 0, []; self.current_yoga_pose_index = 0
        self.rhythm_sequence, self.rhythm_current_step = [], 0; self.dodger_objects, self.dodger_lives = [], 0
        self.ninja_slicer_objects, self.ninja_slicer_lives = [], 0
        self.left_hand_trail, self.right_hand_trail = deque(maxlen=TRAIL_LENGTH), deque(maxlen=TRAIL_LENGTH)
        self.left_hand_rect, self.right_hand_rect = None, None
        self.session_nuts_collected = 0
        self.session_apples_hit = 0
        self.display_snapshots = []
        for key in self.pose_start_time: self.pose_start_time[key] = None
        # --- Confetti Reset ---
        self.confetti_particles = []
        self.confetti_triggered_this_session = False
        
    def reset_cycle(self):
        mode_reset_map = {
            'random': self.reset_random_mode_cycle, 
            'yoga': self.reset_yoga_mode_cycle, 
            'rhythm': self.reset_rhythm_mode_cycle, 
            'dodger': self.reset_dodger_mode_cycle,
            'ninja_slicer': self.reset_ninja_slicer_mode_cycle
        }
        reset_func = mode_reset_map.get(self.current_game_mode)
        if reset_func: reset_func()

    def reset_random_mode_cycle(self):
        if self.frozen_box: self.target_points=generate_random_pattern(self.frozen_box,TARGETS_PER_CYCLE); self.matched_flags=[False]*len(self.target_points); self.cycle_start_time=time.time(); self.game_state=self.STATE_PLAYING_CYCLE

    def reset_yoga_mode_cycle(self):
        if self.current_yoga_pose_index < len(self.yoga_landmarks_list):
            self.game_state = self.STATE_PLAYING_CYCLE
            self.cycle_start_time = time.time()
            self.yoga_pose_match_start_time = None
            self.yoga_highest_similarity_score = 0.0
        else:
            self.game_state = self.STATE_SESSION_COMPLETE
        
    def reset_rhythm_mode_cycle(self):
        if self.frozen_box: self.rhythm_sequence=generate_random_pattern(self.frozen_box,RHYTHM_SEQUENCE_LENGTH); self.rhythm_current_step=0; self.score=0; self.combo=1; self.rhythm_target_start_time=time.time(); self.game_state=self.STATE_PLAYING_RHYTHM

    def reset_dodger_mode_cycle(self):
        self.dodger_lives=DODGER_INITIAL_LIVES
        self.score=0
        self.dodger_start_time=time.time()
        self.dodger_objects=[]
        self.dodger_spawn_timer=time.time()
        self.dodger_current_spawn_rate = DODGER_SPAWN_RATE
        self.dodger_last_difficulty_increase = time.time()
        self.game_state=self.STATE_PLAYING_DODGER
        self.session_nuts_collected = 0
        self.session_apples_hit = 0

    def reset_ninja_slicer_mode_cycle(self):
        self.ninja_slicer_lives = NINJA_SLICER_INITIAL_LIVES
        self.score = 0
        self.ninja_slicer_objects = []
        self.ninja_slicer_spawn_timer = time.time()
        self.left_hand_trail.clear()
        self.right_hand_trail.clear()
        self.game_state = self.STATE_PLAYING_NINJA_SLICER

    def add_score(self, points): self.score += points * self.combo

    def trigger_feedback(self, text): self.feedback_text, self.feedback_text_alpha = text, 255

    def handle_pose_trigger(self, pose_detected, pose_key, action_function):
        if pose_detected:
            if self.pose_start_time[pose_key] is None:
                self.pose_start_time[pose_key] = time.time()
            if time.time() - self.pose_start_time[pose_key] >= 1.0:
                action_function()
                # Reset all timers after an action is triggered to prevent conflicts
                for key in self.pose_start_time:
                    self.pose_start_time[key] = None
                return True
        else:
            # If the pose is no longer detected, reset its specific timer
            self.pose_start_time[pose_key] = None
        return False

    def draw_rounded_rect(self, img, top_left, bottom_right, color, alpha, radius=20, border_color=None, border_thickness=4, fill=True):
        x1, y1 = top_left; x2, y2 = bottom_right; overlay = img.copy()
        
        if fill:
            cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
            cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
            cv2.ellipse(overlay, (x1+radius, y1+radius), (radius,radius), 180, 0, 90, color, -1)
            cv2.ellipse(overlay, (x2-radius, y1+radius), (radius,radius), 270, 0, 90, color, -1)
            cv2.ellipse(overlay, (x1+radius, y2-radius), (radius,radius), 90, 0, 90, color, -1)
            cv2.ellipse(overlay, (x2-radius, y2-radius), (radius,radius), 0, 0, 90, color, -1)
        
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        if border_color:
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), border_color, border_thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), border_color, border_thickness)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), border_color, border_thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), border_color, border_thickness)
            
            cv2.ellipse(img, (x1+radius, y1+radius), (radius,radius), 180, 0, 90, border_color, border_thickness)
            cv2.ellipse(img, (x2-radius, y1+radius), (radius,radius), 270, 0, 90, border_color, border_thickness)
            cv2.ellipse(img, (x1+radius, y2-radius), (radius,radius), 90, 0, 90, border_color, border_thickness)
            cv2.ellipse(img, (x2-radius, y2-radius), (radius,radius), 0, 0, 90, border_color, border_thickness)

    def draw_hud(self, frame):
        self.draw_rounded_rect(frame, (10, 10), (frame.shape[1] - 10, 80), (0,0,0), 0.4, fill=True)
        score_text = f"{self.current_translation['score']}: {self.score}"
        
        self.draw_text(frame, score_text, (30, 55-15), 30, (0, 255, 255))

        if self.current_game_mode == 'dodger' or self.current_game_mode == 'ninja_slicer':
            lives = self.dodger_lives if self.current_game_mode == 'dodger' else self.ninja_slicer_lives
            info_text = f"{self.current_translation['lives']}: {lives}"
        elif self.current_game_mode == 'random':
            info_text = f"{self.current_translation['zyklen']}: {self.successful_cycles}/{self.total_cycles_attempted}"
        elif self.current_game_mode == 'rhythm':
             info_text = f"{self.current_translation['combo']}: x{self.combo}"
        else:
            info_text = ""

        self.draw_text(frame, info_text, (frame.shape[1] - 30, 55 - 15), 30, (255, 255, 255), align="right")
        
        if self.current_game_mode in ['random', 'yoga']:
            time_limit = CYCLE_TIME_LIMIT if self.current_game_mode == 'random' else YOGA_CYCLE_TIME_LIMIT
            if self.cycle_start_time:
                time_left = max(0, int(time_limit - (time.time() - self.cycle_start_time)))
                time_text = f"{self.current_translation['time']}: {time_left}s"
                self.draw_text(frame, time_text, ((frame.shape[1] // 2) - 150, 55 - 30), 30, (255,255,255))
                bar_w = int(300 * (time_left / time_limit)); 
                cv2.rectangle(frame, ((frame.shape[1] // 2) - 150, 60), ((frame.shape[1] // 2) + 150, 70), (80,80,80), -1); 
                cv2.rectangle(frame, ((frame.shape[1] // 2) - 150, 60), ((frame.shape[1] // 2) - 150 + bar_w, 70), (0,255,255), -1)

    def draw_feedback_text(self, frame):
        if self.feedback_text_alpha > 0:
            font_size = 75
            pos_x = frame.shape[1] // 2
            pos_y = frame.shape[0] // 2
            self.draw_text(frame, self.feedback_text, (pos_x, pos_y), font_size, 
                           (0, 255, 255), align="center", alpha=self.feedback_text_alpha)
            self.feedback_text_alpha -= 10 

    def draw_menu_option(self, frame, text, y_offset, is_highlighted, countdown_time=None):
        text_size = (800, 70) 
        x_box_start = (frame.shape[1] - text_size[0]) // 2
        
        border_color = (0, 255, 255) if is_highlighted else None
        
        self.draw_rounded_rect(frame, (x_box_start, y_offset), (x_box_start + text_size[0], y_offset + text_size[1]), (0,0,0), 0.5, border_color=border_color)
        
        text_y_pos = y_offset + text_size[1] // 2 - 8
        self.draw_text(frame, text, (frame.shape[1] // 2, text_y_pos), 36, (255,255,255), align="center")

        if countdown_time is not None and countdown_time > 0 and self.font_path:
            countdown_text = f"({int(countdown_time)})"
            font = ImageFont.truetype(self.font_path, 36)
            main_text_width = font.getbbox(text)[2] - font.getbbox(text)[0]
            countdown_x = (frame.shape[1] // 2) + (main_text_width // 2) + 40
            self.draw_text(frame, countdown_text, (countdown_x, text_y_pos), 30, (255,0,0), align="left")

    def draw_relay_status(self, frame):
        y_pos = frame.shape[0] - 80; font_size = 24
        if self.left_relay_on: time_left = max(0, self.relay_on_duration - (time.time() - self.left_relay_timer_start)); status_text, color = f"{self.current_translation['left_relay_on_status']} ({int(time_left)}s left)", (0, 255, 0)
        else: status_text, color = self.current_translation['left_relay_off_status'], (0, 0, 255)
        self.draw_text(frame, status_text, (30, y_pos), font_size, color)
        if self.right_relay_on: time_left = max(0, self.relay_on_duration - (time.time() - self.right_relay_timer_start)); status_text, color = f"{self.current_translation['right_relay_on_status']} ({int(time_left)}s left)", (0, 255, 0)
        else: status_text, color = self.current_translation['right_relay_off_status'], (0, 0, 255)
        self.draw_text(frame, status_text, (30, y_pos + 40), font_size, color)

    def overlay_image(self, frame, assets, x, y, alpha_val=1.0):
        if assets is None: return
        img_bgr, img_alpha, (img_w, img_h) = assets['bgr'], assets['alpha'], assets['size']
        frame_h, frame_w, _ = frame.shape
        x_start, y_start = max(0, x), max(0, y)
        x_end, y_end = min(frame_w, x + img_w), min(frame_h, y + img_h)

        if x_end > x_start and y_end > y_start:
            roi = frame[y_start:y_end, x_start:x_end]
            slice_w, slice_h = x_end - x_start, y_end - y_start
            img_slice_x1, img_slice_y1 = max(0, -x), max(0, -y)
            img_slice_x2, img_slice_y2 = img_slice_x1 + slice_w, img_slice_y1 + slice_h
            img_bgr_slice = img_bgr[img_slice_y1:img_slice_y2, img_slice_x1:img_slice_x2]
            alpha_mask_slice = img_alpha[img_slice_y1:img_slice_y2, img_slice_x1:img_slice_x2]
            alpha_mask = (alpha_mask_slice / 255.0 * alpha_val)[:, :, np.newaxis]
            blended_roi = roi * (1 - alpha_mask) + img_bgr_slice * alpha_mask
            frame[y_start:y_end, x_start:x_end] = blended_roi.astype(np.uint8)

    def trigger_confetti_effect(self, w, h, num_particles=200):
        print(f"DEBUG: Triggering confetti with {num_particles} particles.")
        if self.confetti_triggered_this_session:
            return
        self.confetti_triggered_this_session = True
        
        for _ in range(num_particles):
            x = random.randint(0, w)
            y = random.randint(-h // 2, 0)
            vx = random.randint(-10, 10)
            vy = random.randint(10, 20)
            color = (random.randint(50, 255), random.randint(100, 255), random.randint(100, 255))
            lifespan = random.randint(60, 120)
            self.confetti_particles.append([x, y, vx, vy, color, lifespan])

    def draw_confetti(self, frame):
        if not self.confetti_particles:
            return
            
        for p in self.confetti_particles[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[3] += 0.5 
            p[5] -= 1
            
            cv2.circle(frame, (int(p[0]), int(p[1])), 5, p[4], -1)
            
            if p[5] <= 0:
                self.confetti_particles.remove(p)


    def update_frame(self):
        if self.left_relay_on and (time.time() - self.left_relay_timer_start > self.relay_on_duration):
            self.left_relay_on, self.left_relay_timer_start = False, None
            self.send_command_to_arduino(b'0')
        if self.right_relay_on and (time.time() - self.right_relay_timer_start > self.relay_on_duration):
            self.right_relay_on, self.right_relay_timer_start = False, None
            self.send_command_to_arduino(b'3')

        ret, frame = self.cap.read();
        if not ret: self.timer.stop(); return

        h, w, _ = frame.shape 

        frame = cv2.flip(frame, 1); rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame); landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
        
        display_frame = frame.copy()

        lx, ly, rx, ry = -1, -1, -1, -1 

        if landmarks:
            if landmarks[mp_pose.PoseLandmark.LEFT_WRIST].visibility > 0.5:
                lx, ly = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h)
            if landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].visibility > 0.5:
                rx, ry = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h)

        if self.game_state == self.STATE_MENU:
            self.draw_text(display_frame, self.current_translation['pose_to_play'], (50, 80), 75, (0, 255, 255))
            both_hands_up = is_both_hands_up(landmarks)
            
            box_width, box_height, padding = 250, 80, 400
            screen_center_y = h // 2
            group_base_y = screen_center_y + 50 
            
            lang_en_x, lang_en_y = padding + 25, group_base_y
            lang_en_top_left = (lang_en_x, lang_en_y)
            lang_en_bottom_right = (lang_en_x + box_width, lang_en_y + box_height)

            lang_de_x, lang_de_y = padding + 25, group_base_y + box_height + padding // 2 - 120
            lang_de_top_left = (lang_de_x, lang_de_y)
            lang_de_bottom_right = (lang_de_x + box_width, lang_de_y + box_height)
            
            right_hand_over_en = rx != -1 and ry != -1 and lang_en_top_left[0] < rx < lang_en_bottom_right[0] and lang_en_top_left[1] < ry < lang_en_bottom_right[1]
            right_hand_over_de = rx != -1 and ry != -1 and lang_de_top_left[0] < rx < lang_de_bottom_right[0] and lang_de_top_left[1] < ry < lang_de_bottom_right[1]

            en_border_color = (0, 255, 255) if right_hand_over_en else None
            self.draw_rounded_rect(display_frame, lang_en_top_left, lang_en_bottom_right, (0, 0, 0), 0.5, border_color=en_border_color)
            self.draw_text(display_frame, self.current_translation['english'], (lang_en_x + box_width // 2, lang_en_y + 35), 30, (255, 255, 255), align="center")
            if self.handle_pose_trigger(right_hand_over_en, "language_en", self.go_to_english): return

            de_border_color = (0, 255, 255) if right_hand_over_de else None
            self.draw_rounded_rect(display_frame, lang_de_top_left, lang_de_bottom_right, (0, 0, 0), 0.5, border_color=de_border_color)
            self.draw_text(display_frame, self.current_translation['german'], (lang_de_x + box_width // 2, lang_de_y + 35), 30, (255, 255, 255), align="center")
            if self.handle_pose_trigger(right_hand_over_de, "language_de", self.go_to_german): return
            
            self.draw_text(display_frame, self.current_translation['select_language_prompt'], (lang_en_x, lang_en_y - padding // 4 + 50), 24, (0, 255, 255))

            relay_left_x, relay_left_y = w - box_width - padding - 50, group_base_y 
            relay_left_top_left, relay_left_bottom_right = (relay_left_x, relay_left_y), (relay_left_x + box_width, relay_left_y + box_height)
            
            relay_right_x, relay_right_y = w - box_width - padding - 50, group_base_y + box_height + padding // 2 - 120
            relay_right_top_left, relay_right_bottom_right = (relay_right_x, relay_right_y), (relay_right_x + box_width, relay_right_y + box_height)

            left_hand_over_relay_left = lx != -1 and ly != -1 and relay_left_top_left[0] < lx < relay_left_bottom_right[0] and relay_left_top_left[1] < ly < relay_left_bottom_right[1]
            left_hand_over_relay_right = lx != -1 and ly != -1 and relay_right_top_left[0] < lx < relay_right_bottom_right[0] and relay_right_top_left[1] < ly < relay_right_bottom_right[1]

            left_relay_border_color = (0, 255, 0) if self.left_relay_on else ((0, 255, 255) if left_hand_over_relay_left else None)
            self.draw_rounded_rect(display_frame, relay_left_top_left, relay_left_bottom_right, (0, 0, 0), 0.5, border_color=left_relay_border_color)
            self.draw_text(display_frame, self.current_translation['left_relay'], (relay_left_x + box_width // 2, relay_left_y + 35), 30, (255, 255, 255), align="center")
            if not self.left_relay_on and self.handle_pose_trigger(left_hand_over_relay_left, "relay_left_on", self.activate_left_relay): pass

            right_relay_border_color = (0, 255, 0) if self.right_relay_on else ((0, 255, 255) if left_hand_over_relay_right else None)
            self.draw_rounded_rect(display_frame, relay_right_top_left, relay_right_bottom_right, (0, 0, 0), 0.5, border_color=right_relay_border_color)
            self.draw_text(display_frame, self.current_translation['right_relay'], (relay_right_x + box_width // 2, relay_right_y + 35), 30, (255, 255, 255), align="center")
            if not self.right_relay_on and self.handle_pose_trigger(left_hand_over_relay_right, "relay_right_on", self.activate_right_relay): pass
            
            self.draw_text(display_frame, self.current_translation['test_relays'], (relay_left_x, relay_left_y - padding // 4 + 50), 24, (0, 255, 255))

            if self.handle_pose_trigger(both_hands_up, "start_game", self.go_to_game_mode_selection): return
            countdown_val = POSE_HOLD_TIME - (time.time() - self.pose_start_time["start_game"]) if self.pose_start_time["start_game"] else None
            self.draw_menu_option(display_frame, self.current_translation['start_game'], screen_center_y - 200, both_hands_up, countdown_val)


        elif self.game_state == self.STATE_GAME_MODE_SELECTION:
            SELECTION_COOLDOWN = 2.0  # 2-second delay before a selection can be made
            can_select = self.mode_selection_entry_time is not None and (time.time() - self.mode_selection_entry_time > SELECTION_COOLDOWN)

            self.draw_text(display_frame, self.current_translation['select_game_mode'], (50, 80), 75, (0, 255, 255))
            
            # Show a countdown message if the cooldown is active
            if not can_select:
                time_left = SELECTION_COOLDOWN - (time.time() - self.mode_selection_entry_time)
                self.draw_text(display_frame, f"Ready in {math.ceil(time_left)}...", (display_frame.shape[1] // 2, 200), 40, (255, 255, 0), align="center")

            poses = {
                "random": is_hand_raised(landmarks,'RIGHT'), 
                "yoga": is_hand_raised(landmarks,'LEFT'), 
                "rhythm": is_t_pose(landmarks), 
                "dodger": is_hands_on_hips(landmarks), 
                "stats": is_both_hands_down(landmarks),
                "ninja_slicer": is_hands_folded(landmarks)
            }
            
            # --- POSE CONFLICT RESOLUTION ---
            # Prioritize more specific poses to avoid ambiguity
            if poses["dodger"]: # If hands are on hips
                poses["ninja_slicer"] = False # They can't also be folded
            
            is_multi_pose = poses["rhythm"] or poses["dodger"] or poses["stats"] or poses["ninja_slicer"]
            if is_multi_pose: 
                poses["random"], poses["yoga"] = False, False
            
            options = [
                (self.current_translation['yoga_mode'], poses["yoga"], "yoga_mode", lambda: self.start_game_mode('yoga')), 
                (self.current_translation['collector_mode'], poses["dodger"], "dodger_mode", lambda: self.start_game_mode('dodger')),
                (self.current_translation['ninja_slicer_mode'], poses["ninja_slicer"], "ninja_slicer_mode", lambda: self.start_game_mode('ninja_slicer')),
                (self.current_translation['rhythm_mode'], poses["rhythm"], "rhythm_mode", lambda: self.start_game_mode('rhythm')), 
                (self.current_translation['random_mode'], poses["random"], "random_mode", lambda: self.start_game_mode('random'))
            ]
            
            y_pos = 250
            for text, detected, key, action in options:
                if can_select:
                    if self.handle_pose_trigger(detected, key, action): return

                is_highlighted = detected and can_select
                countdown = POSE_HOLD_TIME - (time.time() - self.pose_start_time[key]) if self.pose_start_time[key] is not None and can_select else None
                self.draw_menu_option(display_frame, text, y_pos, is_highlighted, countdown)
                y_pos += 90

        elif self.game_state == self.STATE_STATS:
            self.draw_rounded_rect(display_frame, (50, 50), (display_frame.shape[1]-50, display_frame.shape[0]-150), (0,0,0), 0.7, fill=True)
            self.draw_text(display_frame, self.current_translation['player_stats'], (100, 120), 45, (0, 255, 255))
            y_pos = 200; font_size = 30
            for mode, score in self.stats['high_scores'].items():
                translated_mode = self.current_translation.get(f"mode_{mode}", mode)
                self.draw_text(display_frame, f"{self.current_translation['high_score']} ({translated_mode}): {score}", (100, y_pos), font_size, (255,255,255))
                y_pos += 50
            y_pos += 20
            for mode, count in self.stats['games_played'].items():
                translated_mode = self.current_translation.get(f"mode_{mode}", mode)
                self.draw_text(display_frame, f"{self.current_translation['games_played']} ({translated_mode}): {count}", (100, y_pos), font_size, (255,255,255))
                y_pos += 50
            self.draw_text(display_frame, f"{self.current_translation['total_nuts_collected']}: {self.stats.get('total_nuts_collected', 0)}", (100, y_pos + 20), font_size, (0,255,0))
            self.draw_text(display_frame, self.current_translation['back_to_menu'], (100, display_frame.shape[0]-80), font_size, (0,255,255))
            if self.handle_pose_trigger(is_both_hands_down(landmarks), "back_to_main_from_mode_select", self.go_to_game_mode_selection): return

        elif self.game_state == self.STATE_WAITING_FOR_PLAYER:
            if landmarks:
                mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            self.draw_text(display_frame, self.current_translation['stand_center'], (display_frame.shape[1] // 2, display_frame.shape[0]//2), 36, (255,255,255), align="center")
            if landmarks and is_full_body_visible(landmarks):
                self.frozen_box = get_bounding_box(landmarks, frame.shape[:2])
                if self.frozen_box: self.bounding_box_frozen = True; self.reset_cycle()

        elif self.game_state == self.STATE_IN_GAME_MENU:
            display_frame = cv2.GaussianBlur(display_frame, (99,99), 0)
            self.draw_text(display_frame, self.current_translation['game_paused'], (50, 80), 60, (255,255,0))
            menu_options = [(self.current_translation['resume'], lambda: is_hand_raised(landmarks, 'LEFT'), "resume_game", self.resume_game), (self.current_translation['main_menu'], lambda: is_hand_raised(landmarks, 'RIGHT'), "ingame_main_menu", self.go_to_main_menu_from_ingame)]
            y_pos = 200
            for text, pose_check, key, action in menu_options:
                countdown = POSE_HOLD_TIME - (time.time() - self.pose_start_time[key]) if self.pose_start_time[key] else None
                if self.handle_pose_trigger(pose_check(), key, action): return
                self.draw_menu_option(display_frame, text, y_pos, self.pose_start_time[key] is not None, countdown)
                y_pos += 100

        elif self.game_state == self.STATE_PLAYING_CYCLE:
            if self.current_game_mode == 'random': self.run_random_mode_frame(display_frame, landmarks, rgb_frame)
            elif self.current_game_mode == 'yoga': self.run_yoga_mode_frame(display_frame, landmarks, rgb_frame)

        elif self.game_state == self.STATE_PLAYING_RHYTHM: self.run_rhythm_game_frame(display_frame, landmarks, rgb_frame)
        elif self.game_state == self.STATE_PLAYING_DODGER: self.run_dodger_game_frame(display_frame, landmarks, rgb_frame)
        elif self.game_state == self.STATE_PLAYING_NINJA_SLICER: self.run_ninja_slicer_game_frame(display_frame, landmarks, rgb_frame)

        elif self.game_state == self.STATE_CYCLE_END_FEEDBACK:
            if time.time() - self.feedback_display_start_time > FEEDBACK_DISPLAY_DURATION:
                if self.current_game_mode == 'random' and self.total_cycles_attempted >= TARGET_CYCLES:
                    self.game_state = self.STATE_SESSION_COMPLETE
                elif self.current_game_mode == 'yoga' and self.current_yoga_pose_index >= len(self.yoga_landmarks_list):
                    self.game_state = self.STATE_SESSION_COMPLETE
                else:
                    self.reset_cycle()
            else:
                color = (0,0,255) if "MISSED" in self.cycle_end_message or "VERPASST" in self.cycle_end_message else (0,255,0)
                self.draw_text(display_frame, self.cycle_end_message, (int(display_frame.shape[1]*0.5), int(display_frame.shape[0]*0.5)), 60, color, align="center")

        elif self.game_state == self.STATE_SESSION_COMPLETE:
            if self.last_session_end_time is None:
                self.last_session_end_time = time.time()

                should_celebrate = False
                mode_map = {'dodger': 'Collector', 'random': 'Random', 'rhythm': 'Rhythm', 'yoga': 'Yoga', 'ninja_slicer': 'NinjaSlicer'}
                mode_name = mode_map.get(self.current_game_mode)

                if mode_name:
                    previous_high_score = self.stats["high_scores"].get(mode_name, 0)
                    print(f"DEBUG: Final Score: {self.score}, High Score for {mode_name}: {previous_high_score}")

                    if self.score > previous_high_score:
                        should_celebrate = True
                        print("DEBUG: New high score!")

                    if not should_celebrate:
                        if self.current_game_mode == 'random' and self.successful_cycles > 0 and self.successful_cycles == self.total_cycles_attempted:
                            should_celebrate = True
                            print("DEBUG: Perfect 'Random' mode run!")
                        elif self.current_game_mode == 'yoga' and self.score > 0 and self.score == len(self.yoga_landmarks_list):
                            should_celebrate = True
                            print("DEBUG: Perfect 'Yoga' mode run!")

                if should_celebrate:
                    self.trigger_confetti_effect(w, h)
                    celebrate_sound.play()
                
                self.update_leaderboard_and_stats()

                # --- NEW SNAPSHOT SELECTION LOGIC ---
                if len(self.session_snapshots) > 3:
                    self.display_snapshots = random.sample(self.session_snapshots, 3)
                else:
                    self.display_snapshots = self.session_snapshots

            
            title_text = self.current_translation['game_over'] if self.current_game_mode in ['dodger', 'rhythm', 'ninja_slicer'] else self.current_translation['session_complete']
            self.draw_text(display_frame, title_text, (50, 100), 60, (0, 255, 0))

            if self.current_game_mode == 'dodger':
                stats_text = f"{self.current_translation['final_score']}: {self.score} | {self.current_translation['nuts_collected']}: {self.session_nuts_collected}"
            elif self.current_game_mode == 'random':
                stats_text = f"{self.current_translation['zyklen']}: {self.successful_cycles}/{self.total_cycles_attempted}"
            elif self.current_game_mode == 'yoga':
                stats_text = f"{self.current_translation['poses_completed']}: {self.score}"
            else: # Covers rhythm and ninja_slicer
                stats_text = f"{self.current_translation['final_score']}: {self.score}"
            self.draw_text(display_frame, stats_text, (50, 180), 45, (255, 255, 255))

            mode_map = {'dodger': 'Collector', 'random': 'Random', 'rhythm': 'Rhythm', 'yoga': 'Yoga', 'ninja_slicer': 'NinjaSlicer'}
            mode_name = mode_map.get(self.current_game_mode)
            if mode_name:
                high_score = self.stats['high_scores'].get(mode_name, 0)
                translated_mode_name = self.current_translation.get(f"mode_{mode_name}", mode_name)
                high_score_text = f"{self.current_translation['high_score']} ({translated_mode_name}): {high_score}"
                self.draw_text(display_frame, high_score_text, (50, 240), 36, (0, 255, 255))
            
            y_pos, x_pos, margin = 320, 50, 20
            # --- MODIFIED DISPLAY LOOP ---
            for i, (snap, success) in enumerate(self.display_snapshots):
                thumb = cv2.resize(cv2.cvtColor(snap, cv2.COLOR_RGB2BGR), (150, 150))
                border_color = (0, 255, 0) if success else (0, 0, 255)
                cv2.rectangle(thumb, (0, 0), (149, 149), border_color, 4)
                if x_pos + 150 > display_frame.shape[1]:
                    x_pos, y_pos = 50, y_pos + 150 + margin
                if y_pos + 150 < display_frame.shape[0]:
                    display_frame[y_pos:y_pos+150, x_pos:x_pos+150] = thumb
                x_pos += 150 + margin
            if time.time() - self.last_session_end_time > SESSION_COOLDOWN:
                box_width, box_height, padding = 250, 80, 150
                screen_center_y_for_relays = h // 2 + 50
                
                relay_left_end_x = w // 2 - box_width - padding // 2
                relay_left_end_y = screen_center_y_for_relays
                relay_left_end_top_left = (relay_left_end_x, relay_left_end_y)
                relay_left_end_bottom_right = (relay_left_end_x + box_width, relay_left_end_y + box_height)

                relay_right_end_x = w // 2 + padding // 2
                relay_right_end_y = screen_center_y_for_relays
                relay_right_end_top_left = (relay_right_end_x, relay_right_end_y)
                relay_right_end_bottom_right = (relay_right_end_x + box_width, relay_right_end_y + box_height)
                
                right_hand_over_relay_left_end = rx != -1 and ry != -1 and relay_left_end_top_left[0] < rx < relay_left_end_bottom_right[0] and relay_left_end_top_left[1] < ry < relay_left_end_bottom_right[1]
                left_hand_over_relay_right_end = lx != -1 and ly != -1 and relay_right_end_top_left[0] < lx < relay_right_end_bottom_right[0] and relay_right_end_top_left[1] < ly < relay_right_end_bottom_right[1]

                self.draw_text(display_frame, self.current_translation['test_relays_end'], (relay_left_end_x + box_width // 2 , relay_left_end_y  - 50), 24, (0, 255, 255))

                left_relay_end_border_color = (0, 255, 0) if self.left_relay_on else ((0, 255, 255) if right_hand_over_relay_left_end else None)
                self.draw_rounded_rect(display_frame, relay_left_end_top_left, relay_left_end_bottom_right, (0, 0, 0), 0.5, border_color=left_relay_end_border_color)
                self.draw_text(display_frame, self.current_translation['left_relay'], (relay_left_end_x + box_width // 2, relay_left_end_y + 35), 30, (255, 255, 255), align="center")
                if not self.left_relay_on and self.handle_pose_trigger(right_hand_over_relay_left_end, "relay_left_on_end_screen", self.activate_left_relay): pass

                right_relay_end_border_color = (0, 255, 0) if self.right_relay_on else ((0, 255, 255) if left_hand_over_relay_right_end else None)
                self.draw_rounded_rect(display_frame, relay_right_end_top_left, relay_right_end_bottom_right, (0, 0, 0), 0.5, border_color=right_relay_end_border_color)
                self.draw_text(display_frame, self.current_translation['right_relay'], (relay_right_end_x + box_width // 2, relay_right_end_y + 35), 30, (255, 255, 255), align="center")
                if not self.right_relay_on and self.handle_pose_trigger(left_hand_over_relay_right_end, "relay_right_on_end_screen", self.activate_right_relay): pass
                
           
                if self.handle_pose_trigger(is_both_hands_up(landmarks), "restart_session", self.go_to_main_menu_from_ingame): return
                self.draw_menu_option(display_frame, self.current_translation['both_hands_up_menu'], h - 100, is_both_hands_up(landmarks))

        
        mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if self.game_state in [self.STATE_PLAYING_CYCLE, self.STATE_PLAYING_RHYTHM, self.STATE_PLAYING_DODGER, self.STATE_PLAYING_NINJA_SLICER]:
            self.draw_hud(display_frame)
        self.draw_feedback_text(display_frame)
        self.draw_relay_status(display_frame)
        
        self.draw_confetti(display_frame)

        h, w, ch = display_frame.shape; qt_img = QtGui.QImage(display_frame.data, w, h, ch*w, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_img.scaled(self.video_label.width(), self.video_label.height(), QtCore.Qt.KeepAspectRatio)))

    def run_gameplay_common_logic(self, frame, landmarks):
        if not self.bounding_box_frozen: self.game_state = self.STATE_WAITING_FOR_PLAYER; return False
        if self.handle_pose_trigger(landmarks is None, "no_pose_detected_pause", self.pause_game): return False
        return True

    def run_random_mode_frame(self, frame, landmarks, rgb_frame):
        if not self.run_gameplay_common_logic(frame, landmarks): return
        self.matched_flags = match_points_to_keypoints(self.target_points, landmarks, frame.shape[:2])
        all_matched = all(self.matched_flags)
        for i, (tx, ty) in enumerate(self.target_points): cv2.circle(frame, (tx,ty), TARGET_RADIUS, (0,255,0) if self.matched_flags[i] else (0,0,255), -1)
        if time.time() - self.cycle_start_time > CYCLE_TIME_LIMIT or all_matched:
            self.total_cycles_attempted += 1; success = all_matched and (time.time() - self.cycle_start_time <= CYCLE_TIME_LIMIT)
            if success: self.successful_cycles += 1; self.cycle_end_message = self.current_translation['success']; match_sound.play()
            else: self.cycle_end_message = self.current_translation['missed']; unmatch_sound.play()
            self.session_snapshots.append((rgb_frame, success)); self.feedback_display_start_time = time.time(); self.game_state = self.STATE_CYCLE_END_FEEDBACK

    def run_yoga_mode_frame(self, frame, landmarks, rgb_frame):
        if not self.run_gameplay_common_logic(frame, landmarks): return
        if self.current_yoga_pose_index >= len(self.yoga_landmarks_list): self.game_state = self.STATE_SESSION_COMPLETE; return
        ref_img = self.yoga_images[self.current_yoga_pose_index]; ref_h, ref_w, _ = ref_img.shape
        self.draw_rounded_rect(frame, (frame.shape[1]-ref_h-30, frame.shape[0]-ref_h-30), (frame.shape[1]-20, frame.shape[0]-20), (0,0,0), 0.5, fill=True)
        frame[frame.shape[0]-ref_h-25:frame.shape[0]-25, frame.shape[1]-ref_w-25:frame.shape[1]-25] = ref_img
        success = False
        if landmarks:
            h, w, _ = frame.shape
            current_lm = np.array([[lm.x * w, lm.y * h] for lm in landmarks]); similarity = calculate_similarity(current_lm, self.yoga_landmarks_list[self.current_yoga_pose_index])
            self.yoga_highest_similarity_score = similarity
            if similarity > YOGA_SIMILARITY_THRESHOLD:
                if self.yoga_pose_match_start_time is None: self.yoga_pose_match_start_time = time.time()
                if time.time() - self.yoga_pose_match_start_time >= YOGA_POSE_HOLD_DURATION:
                    success = True
                    self.score += 1
                    match_sound.play()
            else: self.yoga_pose_match_start_time = None
        if success or (time.time() - self.cycle_start_time > YOGA_CYCLE_TIME_LIMIT):
            self.cycle_end_message = self.current_translation['pose_success'] if success else self.current_translation['pose_missed']; unmatch_sound.play() if not success else None
            self.session_snapshots.append((rgb_frame, success)); self.current_yoga_pose_index += 1
            self.game_state = self.STATE_CYCLE_END_FEEDBACK; self.feedback_display_start_time = time.time()

    def run_rhythm_game_frame(self, frame, landmarks, rgb_frame):
        if not self.run_gameplay_common_logic(frame, landmarks): return
        if not self.rhythm_sequence: self.reset_rhythm_mode_cycle(); return
        time_left = RHYTHM_TARGET_TIME_LIMIT - (time.time() - self.rhythm_target_start_time)
        if time_left < 0:
            if combo_break_sound: combo_break_sound.play(); self.trigger_feedback(self.current_translation['missed']); self.combo = 1
            self.rhythm_current_step += 1; self.rhythm_target_start_time = time.time()
        if self.rhythm_current_step >= len(self.rhythm_sequence):
            self.game_state = self.STATE_SESSION_COMPLETE; return
        tx, ty = self.rhythm_sequence[self.rhythm_current_step]
        cv2.circle(frame, (tx, ty), TARGET_RADIUS, (0, 255, 255), -1)
        if landmarks and match_points_to_keypoints([(tx,ty)], landmarks, frame.shape[:2])[0]:
            if hit_sound: hit_sound.play(); self.add_score(10); self.trigger_feedback(f"{self.current_translation['hit']}! x{self.combo}"); self.combo += 1
            self.session_snapshots.append((rgb_frame, True))
            self.rhythm_current_step += 1; self.rhythm_target_start_time = time.time()
            if self.rhythm_current_step >= len(self.rhythm_sequence):
                self.add_score(100); celebrate_sound.play(); self.game_state = self.STATE_SESSION_COMPLETE

    def run_dodger_game_frame(self, frame, landmarks, rgb_frame):
        if not self.run_gameplay_common_logic(frame, landmarks): return
        
        # --- Difficulty Progression ---
        if time.time() - self.dodger_last_difficulty_increase > 15: # Increase difficulty every 15 seconds
            self.dodger_current_spawn_rate = max(0.2, self.dodger_current_spawn_rate * 0.95) # Faster spawns
            self.dodger_last_difficulty_increase = time.time()
            print(f"INFO: Collector difficulty increased! New spawn rate: {self.dodger_current_spawn_rate:.2f}s")

        # --- Spawning Logic ---
        if time.time() > self.dodger_spawn_timer + self.dodger_current_spawn_rate:
            obj_rect = pygame.Rect(random.randint(0, frame.shape[1]-50), -50, 50, 50)
            # Randomly choose between the three item types
            obj_type = random.choice(['gear', 'nut', 'apple', 'apple']) 
            self.dodger_objects.append((obj_rect, obj_type))
            self.dodger_spawn_timer = time.time()

        # --- Hand Tracking and Magnet Effect ---
        self.left_hand_rect, self.right_hand_rect = None, None
        h, w, _ = frame.shape
        if landmarks and self.magnet_img_assets:
            # Left Hand
            left_wrist_lm = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            if left_wrist_lm.visibility > 0.5:
                lx, ly = int(left_wrist_lm.x * w), int(left_wrist_lm.y * h)
                cursor_w, cursor_h = self.magnet_img_assets['size']
                self.left_hand_rect = pygame.Rect(lx - cursor_w//2, ly - cursor_h//2, cursor_w, cursor_h)
                self.overlay_image(frame, self.magnet_img_assets, self.left_hand_rect.x, self.left_hand_rect.y)
            # Right Hand
            right_wrist_lm = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            if right_wrist_lm.visibility > 0.5:
                rx, ry = int(right_wrist_lm.x * w), int(right_wrist_lm.y * h)
                cursor_w, cursor_h = self.magnet_img_assets['size']
                self.right_hand_rect = pygame.Rect(rx - cursor_w//2, ry - cursor_h//2, cursor_w, cursor_h)
                self.overlay_image(frame, self.magnet_img_assets, self.right_hand_rect.x, self.right_hand_rect.y)

        # --- Object Update, Drawing, and Collision ---
        for obj_rect, obj_type in self.dodger_objects[:]:
            is_attracted = False
            # Apply magnet only to collectible items
            if obj_type in ['gear', 'nut']:
                dist_l, dist_r = float('inf'), float('inf')
                if self.left_hand_rect: dist_l = np.linalg.norm(np.array(self.left_hand_rect.center) - np.array(obj_rect.center))
                if self.right_hand_rect: dist_r = np.linalg.norm(np.array(self.right_hand_rect.center) - np.array(obj_rect.center))
                
                if min(dist_l, dist_r) < DODGER_MAGNET_RANGE:
                    is_attracted = True
                    target_hand = self.left_hand_rect if dist_l < dist_r else self.right_hand_rect
                    vec_x, vec_y = target_hand.centerx - obj_rect.centerx, target_hand.centery - obj_rect.centery
                    obj_rect.x += int(vec_x * DODGER_MAGNET_STRENGTH)
                    obj_rect.y += int(vec_y * DODGER_MAGNET_STRENGTH)
            
            # Gravity
            if not is_attracted: obj_rect.y += 10
            
            # Remove if off-screen
            if obj_rect.top > frame.shape[0]: 
                self.dodger_objects.remove((obj_rect, obj_type))
                continue
            
            # --- Drawing Logic ---
            asset_to_draw = None
            backup_color = (255, 255, 255)
            if obj_type == 'gear':
                asset_to_draw = self.gear_img_assets
                backup_color = (0, 255, 0)
            elif obj_type == 'nut':
                asset_to_draw = self.nuts_img_assets
                backup_color = (0, 255, 0)
            elif obj_type == 'apple':
                asset_to_draw = self.apple_img_assets
                backup_color = (0, 0, 255)

            if asset_to_draw:
                self.overlay_image(frame, asset_to_draw, obj_rect.x, obj_rect.y)
            else: # Fallback rectangle
                cv2.rectangle(frame, (obj_rect.x, obj_rect.y), (obj_rect.right, obj_rect.bottom), backup_color, -1)

            # --- Collision Logic ---
            collided = False
            if self.left_hand_rect and self.left_hand_rect.colliderect(obj_rect): collided = True
            if not collided and self.right_hand_rect and self.right_hand_rect.colliderect(obj_rect): collided = True
            
            if collided:
                # Good item collision
                if obj_type in ['gear', 'nut']:
                    self.score += DODGER_POINTS_PER_ITEM
                    self.session_nuts_collected += 1
                    hit_sound.play()
                    self.trigger_feedback(f"+{DODGER_POINTS_PER_ITEM}!")
                    self.session_snapshots.append((rgb_frame, True))
                # Bad item collision
                elif obj_type == 'apple':
                    combo_break_sound.play()
                    self.dodger_lives -= 1
                    self.session_apples_hit += 1
                    self.trigger_feedback(self.current_translation['ouch'])
                    self.session_snapshots.append((rgb_frame, False))

                self.dodger_objects.remove((obj_rect, obj_type))
                
                # Check for game over
                if self.dodger_lives <= 0:
                    self.trigger_feedback(self.current_translation['game_over'])
                    self.game_state = self.STATE_SESSION_COMPLETE
                    self.last_session_end_time = None
                    break

    def run_ninja_slicer_game_frame(self, frame, landmarks, rgb_frame):
        if not self.run_gameplay_common_logic(frame, landmarks): return
        h, w, _ = frame.shape
        
        # --- Update and Draw Hand Trails ---
        if landmarks:
            if landmarks[mp_pose.PoseLandmark.LEFT_WRIST].visibility > 0.5:
                self.left_hand_trail.append((int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w), int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h)))
            if landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].visibility > 0.5:
                self.right_hand_trail.append((int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h)))
        
        for trail in [self.left_hand_trail, self.right_hand_trail]:
            for i in range(1, len(trail)):
                if trail[i-1] is None or trail[i] is None: continue
                thickness = int(np.sqrt(TRAIL_LENGTH / float(i + 1)) * 2.5)
                cv2.line(frame, trail[i-1], trail[i], (50, 220, 255), thickness)

        # --- Spawn New Objects (Top to Bottom) ---
        if time.time() > self.ninja_slicer_spawn_timer + NINJA_SLICER_SPAWN_RATE:
            obj_type = 'watermelon' if random.random() < NINJA_SLICER_FRUIT_CHANCE else 'bomb'
            asset = self.watermelon_img if obj_type == 'watermelon' else self.bomb_img
            start_x = random.randint(int(w * 0.1), int(w * 0.9))
            start_y = -asset['size'][1] # Start above the screen
            vel_x = random.uniform(-2, 2)
            vel_y = random.uniform(2, 5) # Initial downward velocity
            
            obj = {'rect': pygame.Rect(start_x, start_y, asset['size'][0], asset['size'][1]), 
                   'type': obj_type, 'vel': [vel_x, vel_y], 'asset': asset, 'remove': False, 'lifespan': -1}
            self.ninja_slicer_objects.append(obj)
            self.ninja_slicer_spawn_timer = time.time()

        # --- Update and Draw All Objects ---
        new_objects = []
        for obj in self.ninja_slicer_objects:
            # Update position & apply gravity
            if obj['type'] != 'splash':
                obj['vel'][1] += 0.2 
                obj['rect'].x += int(obj['vel'][0])
                obj['rect'].y += int(obj['vel'][1])

            # Slicing Logic
            if obj['type'] in ['watermelon', 'bomb']:
                was_sliced = False
                for trail in [self.left_hand_trail, self.right_hand_trail]:
                    if len(trail) > 1:
                        # Check collision between the object's rect and the last segment of the trail
                        if obj['rect'].clipline(trail[-2], trail[-1]):
                            was_sliced = True
                            slice_pos = obj['rect'].center
                            break
                
                if was_sliced:
                    obj['remove'] = True
                    if obj['type'] == 'watermelon':
                        slice_sound.play()
                        self.score += NINJA_SLICER_POINTS_PER_FRUIT
                        self.session_snapshots.append((rgb_frame, True))
                        # Create two halves
                        new_objects.append({'rect': pygame.Rect(obj['rect'].x, obj['rect'].y, self.watermelon_half_top_img['size'][0], self.watermelon_half_top_img['size'][1]), 'type': 'half', 'vel': [obj['vel'][0] - 1.5, obj['vel'][1] - 2], 'asset': self.watermelon_half_top_img, 'remove': False, 'lifespan': -1})
                        new_objects.append({'rect': pygame.Rect(obj['rect'].x, obj['rect'].y + self.watermelon_half_top_img['size'][1], self.watermelon_half_bottom_img['size'][0], self.watermelon_half_bottom_img['size'][1]), 'type': 'half', 'vel': [obj['vel'][0] + 1.5, obj['vel'][1] - 1.5], 'asset': self.watermelon_half_bottom_img, 'remove': False, 'lifespan': -1})
                        # Create splash effect
                        new_objects.append({'rect': pygame.Rect(slice_pos[0] - self.splash_img['size'][0]//2, slice_pos[1] - self.splash_img['size'][1]//2, self.splash_img['size'][0], self.splash_img['size'][1]), 'type': 'splash', 'vel': [0,0], 'asset': self.splash_img, 'remove': False, 'lifespan': 10}) # Lifespan in frames
                    else: # Bomb
                        bomb_sound.play()
                        self.ninja_slicer_lives -= 1
                        self.trigger_feedback(self.current_translation['ouch'])
                        self.session_snapshots.append((rgb_frame, False))
                        # Create explosion/splash effect for bomb
                        new_objects.append({'rect': pygame.Rect(slice_pos[0] - self.splash_img['size'][0]//2, slice_pos[1] - self.splash_img['size'][1]//2, self.splash_img['size'][0], self.splash_img['size'][1]), 'type': 'splash', 'vel': [0,0], 'asset': self.splash_img, 'remove': False, 'lifespan': 15})

            # Handle off-screen objects
            if obj['rect'].top > h:
                obj['remove'] = True
                if obj['type'] == 'watermelon': # Missed a fruit
                    self.ninja_slicer_lives -= 1
                    combo_break_sound.play()
                    self.trigger_feedback(self.current_translation['missed'])
            
            # Handle splash lifespan
            if obj['type'] == 'splash':
                obj['lifespan'] -= 1
                if obj['lifespan'] <= 0:
                    obj['remove'] = True

            # Draw the object
            alpha = 1.0
            if obj['type'] == 'splash':
                alpha = obj['lifespan'] / 10.0 # Fade out splash
            self.overlay_image(frame, obj['asset'], obj['rect'].x, obj['rect'].y, alpha)

        # Clean up list and add new objects
        self.ninja_slicer_objects = [obj for obj in self.ninja_slicer_objects if not obj['remove']] + new_objects

        # --- Check for Game Over ---
        if self.ninja_slicer_lives <= 0:
            self.trigger_feedback(self.current_translation['game_over'])
            self.game_state = self.STATE_SESSION_COMPLETE
            self.last_session_end_time = None
            return

    def go_to_game_mode_selection(self):
        self.game_state = self.STATE_GAME_MODE_SELECTION
        self.mode_selection_entry_time = time.time()
        # Reset all pose timers to prevent immediate selection from a carried-over pose.
        for key in self.pose_start_time:
            self.pose_start_time[key] = None
    def go_to_stats_screen(self): self.game_state = self.STATE_STATS
    def go_to_main_menu_from_ingame(self): self.game_state = self.STATE_MENU; self.reset_game_progress()
    def pause_game(self): self.previous_game_state = self.game_state; self.game_state = self.STATE_IN_GAME_MENU; self.pause_start_time = time.time(); print("Game Paused.")
    def resume_game(self):
        if self.previous_game_state is not None and self.pause_start_time is not None:
            pause_duration = time.time() - self.pause_start_time
            if self.current_game_mode in ['random', 'yoga']:
                if self.cycle_start_time: self.cycle_start_time += pause_duration
            elif self.current_game_mode == 'rhythm':
                if self.rhythm_target_start_time: self.rhythm_target_start_time += pause_duration
            elif self.current_game_mode == 'dodger':
                if self.dodger_start_time: self.dodger_start_time += pause_duration
            elif self.current_game_mode == 'ninja_slicer':
                if self.ninja_slicer_spawn_timer: self.ninja_slicer_spawn_timer += pause_duration
            self.game_state = self.previous_game_state; self.previous_game_state = None; self.pause_start_time = None; print(f"Resuming game. Paused for {pause_duration:.2f}s.")
    
    def activate_left_relay(self):
        if not self.left_relay_on:
            self.left_relay_on=True
            self.left_relay_timer_start=time.time()
            self.send_command_to_arduino(b'1')
            print("ACTION: Fan Turned ON")

    def activate_right_relay(self):
        if not self.right_relay_on:
            self.right_relay_on = True
            self.right_relay_timer_start=time.time()
            self.send_command_to_arduino(b'2')
            print("ACTION: Water Turned ON")
    
    def go_to_english(self):
        self.set_language('en')
        self.game_state = self.STATE_MENU

    def go_to_german(self):
        self.set_language('de')
        self.game_state = self.STATE_MENU

    def cleanup(self):
        """A dedicated method for all cleanup operations."""
        print("\n--- Cleaning up... ---")
        if self.serial_port and hasattr(self.serial_port, 'is_open') and self.serial_port.is_open:
            try:
                self.send_command_to_arduino(b'0')
                self.send_command_to_arduino(b'3')
                print("Sent final 'ALL OFF' command to Arduino.")
                time.sleep(0.1) 
            except Exception as e:
                print(f"Error sending final OFF command: {e}")
            self.serial_port.close()
            print("Serial port closed.")
        else:
            print("Serial port was not open or already closed.")
        
        self.cap.release()
        self.timer.stop()
        pose.close()
        pygame.quit()
        print("Resources released.")

    def closeEvent(self, event):
        """This event is called when the window is closed."""
        self.cleanup()
        event.accept()

if __name__ == "__main__":
    import sys
    import signal

    app = QtWidgets.QApplication(sys.argv)
    window = PoseGameWindow()
    window.show()

    def signal_handler(sig, frame):
        print('Signal received, shutting down cleanly.')
        window.cleanup()
        app.quit()

    signal.signal(signal.SIGINT, signal_handler)

    timer = QtCore.QTimer()
    timer.start(100)
    timer.timeout.connect(lambda: None)

    sys.exit(app.exec_())