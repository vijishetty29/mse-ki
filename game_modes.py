import time
import random
import pygame
import numpy as np
import cv2
from config import *
from pose_utils import match_points_to_keypoints, calculate_similarity

class GameModes:
    """
    Manages the specific logic for each game mode.
    This class is responsible for updating the game state, handling player input,
    and determining the outcomes for each of the different games.
    """
    def __init__(self, game_instance, drawing_utils):
        """
        Initializes the GameModes class.
        Args:
            game_instance: The main PoseGameWindow instance.
            drawing_utils: An instance of the DrawingUtils class for rendering.
        """
        self.game = game_instance
        self.drawing_utils = drawing_utils
        # Sounds can be accessed via self.game.sound_name
        self.game.slice_sound = pygame.mixer.Sound("assets/sounds/slice.mp3")
        self.game.bomb_sound = pygame.mixer.Sound("assets/sounds/bomb.mp3")
        self.game.hit_sound = pygame.mixer.Sound("assets/sounds/ding.mp3")
        self.game.combo_break_sound = pygame.mixer.Sound("assets/sounds/unmatch.mp3")
        self.game.celebrate_sound = pygame.mixer.Sound("assets/sounds/celebrate.mp3")
        self.game.match_sound = pygame.mixer.Sound("assets/sounds/ding.mp3")
        self.game.unmatch_sound = pygame.mixer.Sound("assets/sounds/unmatch.mp3")

    def run_cycle_frame(self, frame, landmarks, rgb_frame):
        """Directs to the correct logic for cycle-based games."""
        if self.game.current_game_mode == 'random':
            self.run_random_mode_frame(frame, landmarks, rgb_frame)
        elif self.game.current_game_mode == 'yoga':
            self.run_yoga_mode_frame(frame, landmarks, rgb_frame)

    def run_random_mode_frame(self, frame, landmarks, rgb_frame):
        """Runs a single frame of logic for the 'Random' target-hitting game."""
        if not self.game.game_modes.run_gameplay_common_logic(frame, landmarks): return
        self.game.matched_flags = match_points_to_keypoints(self.game.target_points, landmarks, frame.shape[:2])
        all_matched = all(self.game.matched_flags)
        for i, (tx, ty) in enumerate(self.game.target_points):
            cv2.circle(frame, (tx, ty), TARGET_RADIUS, (0, 255, 0) if self.game.matched_flags[i] else (0, 0, 255), -1)
        
        if time.time() - self.game.cycle_start_time > CYCLE_TIME_LIMIT or all_matched:
            self.game.total_cycles_attempted += 1
            success = all_matched and (time.time() - self.game.cycle_start_time <= CYCLE_TIME_LIMIT)
            if success:
                self.game.successful_cycles += 1
                self.game.cycle_end_message = "SUCCESS!"
                self.game.match_sound.play()
            else:
                self.game.cycle_end_message = "MISSED!"
                self.game.unmatch_sound.play()
            self.game.session_snapshots.append((rgb_frame, success))
            self.game.feedback_display_start_time = time.time()
            self.game.game_state = self.game.STATE_CYCLE_END_FEEDBACK

    def run_yoga_mode_frame(self, frame, landmarks, rgb_frame):
        """Runs a single frame of logic for the 'Yoga' pose matching game."""
        if not self.game.game_modes.run_gameplay_common_logic(frame, landmarks): return
        if self.game.current_yoga_pose_index >= len(self.game.yoga_landmarks_list):
            self.game.game_state = self.game.STATE_SESSION_COMPLETE
            return
        
        ref_img = self.game.yoga_images[self.game.current_yoga_pose_index]
        ref_h, ref_w, _ = ref_img.shape
        self.drawing_utils.draw_rounded_rect(frame, (frame.shape[1]-ref_w-30, frame.shape[0]-ref_h-30), (frame.shape[1]-20, frame.shape[0]-20), (0,0,0), 0.5, fill=True)
        frame[frame.shape[0]-ref_h-25:frame.shape[0]-25, frame.shape[1]-ref_w-25:frame.shape[1]-25] = ref_img

        success = False
        if landmarks:
            h, w, _ = frame.shape
            current_lm = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
            similarity = calculate_similarity(current_lm, self.game.yoga_landmarks_list[self.game.current_yoga_pose_index])
            self.game.yoga_highest_similarity_score = similarity
            if similarity > YOGA_SIMILARITY_THRESHOLD:
                if self.game.yoga_pose_match_start_time is None:
                    self.game.yoga_pose_match_start_time = time.time()
                if time.time() - self.game.yoga_pose_match_start_time >= YOGA_POSE_HOLD_DURATION:
                    success = True
                    self.game.score += 1
                    self.game.match_sound.play()
            else:
                self.game.yoga_pose_match_start_time = None

        if success or (time.time() - self.game.cycle_start_time > YOGA_CYCLE_TIME_LIMIT):
            self.game.cycle_end_message = "POSE SUCCESS!" if success else "POSE MISSED!"
            if not success: self.game.unmatch_sound.play()
            self.game.session_snapshots.append((rgb_frame, success))
            self.game.current_yoga_pose_index += 1
            self.game.game_state = self.game.STATE_CYCLE_END_FEEDBACK
            self.game.feedback_display_start_time = time.time()


    def run_rhythm_game_frame(self, frame, landmarks, rgb_frame):
        """Runs a single frame of logic for the 'Rhythm' game."""
        if not self.game.game_modes.run_gameplay_common_logic(frame, landmarks): return
        if not self.game.rhythm_sequence: self.reset_rhythm_mode_cycle(); return
        
        time_left = RHYTHM_TARGET_TIME_LIMIT - (time.time() - self.game.rhythm_target_start_time)
        if time_left < 0:
            self.game.combo_break_sound.play()
            self.game.trigger_feedback("MISSED!")
            self.game.combo = 1
            self.game.rhythm_current_step += 1
            self.game.rhythm_target_start_time = time.time()

        if self.game.rhythm_current_step >= len(self.game.rhythm_sequence):
            self.game.game_state = self.game.STATE_SESSION_COMPLETE
            return

        tx, ty = self.game.rhythm_sequence[self.game.rhythm_current_step]
        cv2.circle(frame, (tx, ty), TARGET_RADIUS, (0, 255, 255), -1)

        if landmarks and match_points_to_keypoints([(tx, ty)], landmarks, frame.shape[:2])[0]:
            self.game.hit_sound.play()
            self.game.add_score(10)
            self.game.trigger_feedback(f"HIT! x{self.game.combo}")
            self.game.combo += 1
            self.game.session_snapshots.append((rgb_frame, True))
            self.game.rhythm_current_step += 1
            self.game.rhythm_target_start_time = time.time()
            if self.game.rhythm_current_step >= len(self.game.rhythm_sequence):
                self.game.add_score(100)
                self.game.celebrate_sound.play()
                self.game.game_state = self.game.STATE_SESSION_COMPLETE

    def run_dodger_game_frame(self, frame, landmarks, rgb_frame):
        """Runs a single frame of logic for the 'Dodger' game."""
        if not self.game.game_modes.run_gameplay_common_logic(frame, landmarks): return
        
        # Difficulty Progression
        if time.time() - self.game.dodger_last_difficulty_increase > 20:
            self.game.dodger_current_spawn_rate = max(0.15, self.game.dodger_current_spawn_rate * 0.9)
            self.game.dodger_current_good_item_chance = max(0.25, self.game.dodger_current_good_item_chance * 0.95)
            self.game.dodger_last_difficulty_increase = time.time()

        # Spawn new items
        if time.time() > self.game.dodger_spawn_timer + self.game.dodger_current_spawn_rate:
            obj_rect = pygame.Rect(random.randint(0, frame.shape[1]-50), -50, 50, 50)
            obj_type = 'good' if random.random() < self.game.dodger_current_good_item_chance else 'bad'
            self.game.dodger_objects.append((obj_rect, obj_type))
            self.game.dodger_spawn_timer = time.time()
        
        # ... (rest of the logic for updating and drawing dodger items)

    def run_ninja_slicer_game_frame(self, frame, landmarks, rgb_frame):
        """Runs a single frame of logic for the 'Ninja Slicer' game."""
        if not self.game.game_modes.run_gameplay_common_logic(frame, landmarks): return
        h, w, _ = frame.shape
        
        # ... (logic for updating hand trails, spawning, slicing, and drawing objects)

    def run_gameplay_common_logic(self, frame, landmarks):
        """Common logic to run at the start of each gameplay frame."""
        if not self.game.bounding_box_frozen:
            self.game.game_state = self.game.STATE_WAITING_FOR_PLAYER
            return False
        # Pause game logic can be added here
        return True

    def reset_random_mode_cycle(self):
        if self.game.frozen_box:
            self.game.target_points = self.generate_random_pattern(self.game.frozen_box, TARGETS_PER_CYCLE)
            self.game.matched_flags = [False] * len(self.game.target_points)
            self.game.cycle_start_time = time.time()
            self.game.game_state = self.game.STATE_PLAYING_CYCLE

    def reset_yoga_mode_cycle(self):
        if self.game.current_yoga_pose_index < len(self.game.yoga_landmarks_list):
            self.game.game_state = self.game.STATE_PLAYING_CYCLE
            self.game.cycle_start_time = time.time()
            self.game.yoga_pose_match_start_time = None
            self.game.yoga_highest_similarity_score = 0.0
        else:
            self.game.game_state = self.game.STATE_SESSION_COMPLETE

    def reset_rhythm_mode_cycle(self):
        if self.game.frozen_box:
            self.game.rhythm_sequence = self.generate_random_pattern(self.game.frozen_box, RHYTHM_SEQUENCE_LENGTH)
            self.game.rhythm_current_step = 0
            self.game.score = 0
            self.game.combo = 1
            self.game.rhythm_target_start_time = time.time()
            self.game.game_state = self.game.STATE_PLAYING_RHYTHM

    def reset_dodger_mode_cycle(self):
        self.game.dodger_lives = DODGER_INITIAL_LIVES
        self.game.score = 0
        self.game.dodger_start_time = time.time()
        self.game.dodger_objects = []
        self.game.dodger_spawn_timer = time.time()
        self.game.dodger_current_spawn_rate = DODGER_SPAWN_RATE
        self.game.dodger_current_good_item_chance = DODGER_GOOD_ITEM_CHANCE
        self.game.dodger_last_difficulty_increase = time.time()
        self.game.game_state = self.game.STATE_PLAYING_DODGER

    def reset_ninja_slicer_mode_cycle(self):
        self.game.ninja_slicer_lives = NINJA_SLICER_INITIAL_LIVES
        self.game.score = 0
        self.game.ninja_slicer_objects = []
        self.game.ninja_slicer_spawn_timer = time.time()
        self.game.left_hand_trail.clear()
        self.game.right_hand_trail.clear()
        self.game.game_state = self.game.STATE_PLAYING_NINJA_SLICER

    def generate_random_pattern(self, bbox, num_targets):
        x_min, y_min, x_max, y_max = bbox
        margin = 20
        return [(random.randint(x_min - margin, x_max + margin), random.randint(y_min - margin, y_max + margin)) for _ in range(num_targets)]
