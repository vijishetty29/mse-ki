import cv2
import pygame
import time
import mediapipe as mp
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import signal

from config import *
from pose_utils import *
from game_modes import GameModes
from drawing_utils import DrawingUtils

class PoseGameWindow(QtWidgets.QMainWindow):
    STATE_MENU, STATE_GAME_MODE_SELECTION, STATE_IN_GAME_MENU = -3, -2, -1
    STATE_STATS = -5
    STATE_LANGUAGE_SELECTION = -4 
    
    STATE_WAITING_FOR_PLAYER, STATE_PLAYING_CYCLE, STATE_CYCLE_END_FEEDBACK, STATE_SESSION_COMPLETE = 0, 1, 2, 3
    STATE_PLAYING_RHYTHM, STATE_PLAYING_DODGER, STATE_PLAYING_NINJA_SLICER = 4, 5, 6

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Matching Game")
        self.setGeometry(100, 100, 1280, 720)
        self.showFullScreen()

        self.drawing_utils = DrawingUtils()
        self.game_modes = GameModes(self)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.video_label = QtWidgets.QLabel("Waiting...")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        self.game_state = self.STATE_MENU
        self.current_game_mode = None
        self.previous_game_state = None
        self.bounding_box_frozen = False
        self.frozen_box = None
        self.last_session_end_time = None

        self.score = 0
        self.combo = 1
        self.feedback_text = ""
        self.feedback_text_alpha = 0
        self.feedback_display_start_time = 0
        self.mode_selection_entry_time = None

        self.target_points, self.matched_flags, self.successful_cycles, self.total_cycles_attempted, self.cycle_start_time, self.cycle_end_message = [], [], 0, 0, None, ""
        self.rhythm_sequence, self.rhythm_current_step, self.rhythm_target_start_time = [], 0, 0
        self.dodger_objects, self.dodger_lives, self.dodger_start_time, self.dodger_spawn_timer = [], 0, 0, 0
        self.ninja_slicer_objects, self.ninja_slicer_lives, self.ninja_slicer_spawn_timer = [], 0, 0
        self.left_hand_trail, self.right_hand_trail = [], []
        self.left_hand_rect, self.right_hand_rect = None, None
        
        self.session_snapshots = []
        self.display_snapshots = []
        self.pause_start_time = None
        
        self.confetti_particles = []
        self.confetti_triggered_this_session = False
        
        self.pose_start_time = {k: None for k in ["start_game", "random_mode", "yoga_mode", "rhythm_mode", "dodger_mode", "ninja_slicer_mode", "back_to_main_from_mode_select", "resume_game", "ingame_main_menu", "ingame_exit", "t_pose_menu", "restart_session", "no_pose_detected_pause", "view_stats", "select_language", "language_en", "language_de"]}

        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        print("Game Initialized.")

    def start_game_mode(self, mode):
        self.reset_game_progress()
        self.current_game_mode = mode
        if mode in ['random', 'yoga', 'rhythm', 'dodger', 'ninja_slicer']:
            self.game_state = self.STATE_WAITING_FOR_PLAYER
            print(f"Starting {mode.title()} Mode...")

    def reset_game_progress(self):
        self.score, self.combo, self.feedback_text_alpha = 0, 1, 0
        self.bounding_box_frozen, self.frozen_box, self.last_session_end_time = False, None, None
        self.successful_cycles, self.total_cycles_attempted, self.session_snapshots, self.display_snapshots = 0, 0, [], []
        self.rhythm_sequence, self.rhythm_current_step = [], 0
        self.dodger_objects, self.dodger_lives = [], 0
        self.ninja_slicer_objects, self.ninja_slicer_lives = [], 0
        self.left_hand_trail, self.right_hand_trail = [], []
        self.left_hand_rect, self.right_hand_rect = None, None
        for key in self.pose_start_time:
            self.pose_start_time[key] = None
        self.confetti_particles = []
        self.confetti_triggered_this_session = False
        
    def reset_cycle(self):
        mode_reset_map = {
            'random': self.game_modes.reset_random_mode_cycle, 
            'yoga': self.game_modes.reset_yoga_mode_cycle, 
            'rhythm': self.game_modes.reset_rhythm_mode_cycle, 
            'dodger': self.game_modes.reset_dodger_mode_cycle,
            'ninja_slicer': self.game_modes.reset_ninja_slicer_mode_cycle
        }
        reset_func = mode_reset_map.get(self.current_game_mode)
        if reset_func:
            reset_func()

    def add_score(self, points):
        self.score += points * self.combo

    def trigger_feedback(self, text):
        self.feedback_text, self.feedback_text_alpha = text, 255

    def handle_pose_trigger(self, pose_detected, pose_key, action_function):
        if pose_detected:
            if self.pose_start_time[pose_key] is None:
                self.pose_start_time[pose_key] = time.time()
            if time.time() - self.pose_start_time[pose_key] >= POSE_HOLD_TIME:
                action_function()
                for key in self.pose_start_time:
                    self.pose_start_time[key] = None
                return True
        else:
            self.pose_start_time[pose_key] = None
        return False

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        h, w, _ = frame.shape 
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
        
        display_frame = frame.copy()

        if self.game_state == self.STATE_MENU:
            self.drawing_utils.draw_main_menu(self, display_frame, landmarks)
        elif self.game_state == self.STATE_GAME_MODE_SELECTION:
            self.drawing_utils.draw_game_mode_selection(self, display_frame, landmarks)
        elif self.game_state == self.STATE_WAITING_FOR_PLAYER:
            if landmarks:
                mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            self.drawing_utils.draw_text(display_frame, "Stand in the center of the frame!", (w // 2, h // 2), 36, (255,255,255), align="center")
            if landmarks and is_full_body_visible(landmarks):
                self.frozen_box = get_bounding_box(landmarks, frame.shape[:2])
                if self.frozen_box:
                    self.bounding_box_frozen = True
                    self.reset_cycle()
        elif self.game_state in [self.STATE_PLAYING_CYCLE, self.STATE_PLAYING_RHYTHM, self.STATE_PLAYING_DODGER, self.STATE_PLAYING_NINJA_SLICER]:
            game_mode_map = {
                self.STATE_PLAYING_CYCLE: self.game_modes.run_cycle_frame,
                self.STATE_PLAYING_RHYTHM: self.game_modes.run_rhythm_game_frame,
                self.STATE_PLAYING_DODGER: self.game_modes.run_dodger_game_frame,
                self.STATE_PLAYING_NINJA_SLICER: self.game_modes.run_ninja_slicer_game_frame
            }
            game_mode_map[self.game_state](display_frame, landmarks, rgb_frame)
        elif self.game_state == self.STATE_SESSION_COMPLETE:
             self.drawing_utils.draw_session_complete(self, display_frame, landmarks)

        if self.game_state not in [self.STATE_MENU, self.STATE_GAME_MODE_SELECTION]:
             self.drawing_utils.draw_hud(self, display_frame)
        
        self.drawing_utils.draw_feedback_text(self, display_frame)

        h, w, ch = display_frame.shape
        qt_img = QtGui.QImage(display_frame.data, w, h, ch * w, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_img.scaled(self.video_label.width(), self.video_label.height(), QtCore.Qt.KeepAspectRatio)))

    def go_to_game_mode_selection(self):
        self.game_state = self.STATE_GAME_MODE_SELECTION
        self.mode_selection_entry_time = time.time()
        for key in self.pose_start_time:
            self.pose_start_time[key] = None

    def cleanup(self):
        print("\n--- Cleaning up... ---")
        self.cap.release()
        self.timer.stop()
        pose.close()
        pygame.quit()
        print("Resources released.")

    def closeEvent(self, event):
        self.cleanup()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PoseGameWindow()
    window.show()

    def signal_handler(sig, frame):
        print('Signal received, shutting down cleanly.')
        window.cleanup()
        app.quit()

    signal.signal(signal.SIGINT, signal_handler)

    sys.exit(app.exec_())
