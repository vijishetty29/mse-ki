import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import random
import json
from config import *
from pose_utils import *

class DrawingUtils:
    """
    Handles all drawing operations for the game, including text, shapes, images,
    and UI elements like menus and HUDs.
    """
    def __init__(self):
        """Initializes the DrawingUtils, loading fonts, assets, and translations."""
        self.font_path = self.find_font()
        self.translations = self.load_translations()
        self.current_language = 'en'
        self.current_translation = self.translations[self.current_language]
        
        # Load all image assets required for the games
        self.watermelon_img = self.load_image_asset("assets/images/watermelon.png", (80, 80))
        self.watermelon_half_top_img = self.load_image_asset("assets/images/watermelon_half_top.png", (80, 45))
        self.watermelon_half_bottom_img = self.load_image_asset("assets/images/watermelon_half_bottom.png", (80, 45))
        self.bomb_img = self.load_image_asset("assets/images/bomb.png", (70, 70))
        self.splash_img = self.load_image_asset("assets/images/splash.png", (100, 100))
        self.nuts_img_assets = self.load_image_asset("assets/images/gear.png", (50, 50))
        self.apple_img_assets = self.load_image_asset("assets/images/apple.png", (50, 50))

    def find_font(self):
        """Finds a usable font file from a list of possible paths."""
        font_path = "assets/fonts/arial.ttf" 
        try:
            ImageFont.truetype(font_path, 10)
            print(f"INFO: Using font: {font_path}")
            return font_path
        except IOError:
            print(f"WARNING: Could not find font at {font_path}.")
            return None

    def load_translations(self):
        """Loads language translations for UI text."""
        return {
            'en': { 'start_game': "Start Game (Both Hands Up)", 'select_game_mode': "Select Game Mode", 'random_mode': "Random (L Hand)", 'yoga_mode': "Yoga (R Hand)", 'rhythm_mode': "Rhythm (T-Pose)", 'collector_mode': "Collector (Hips)", 'ninja_slicer_mode': "Ninja Slicer (Hands Folded)"},
            'de': { 'start_game': "Spiel starten (Beide Hände hoch)", 'select_game_mode': "Spielmodus auswählen", 'random_mode': "Zufall (L Hand)", 'yoga_mode': "Yoga (R Hand)", 'rhythm_mode': "Rhythmus (T-Pose)", 'collector_mode': "Sammler (Hüften)", 'ninja_slicer_mode': "Ninja Slicer (Hände gefaltet)"}
        }

    def load_image_asset(self, path, size):
        """Loads an image asset from a file path and resizes it."""
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
            placeholder_bgr = np.zeros((size[1], size[0], 3), np.uint8)
            cv2.putText(placeholder_bgr, "?", (size[0]//4, size[1]*3//4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            placeholder_alpha = np.ones((size[1], size[0]), dtype=np.uint8) * 255
            return {'bgr': placeholder_bgr, 'alpha': placeholder_alpha, 'size': size}

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
            info_text = f"{self.current_translation['cycles']}: {self.successful_cycles}/{self.total_cycles_attempted}"
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