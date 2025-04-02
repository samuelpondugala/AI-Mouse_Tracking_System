import cv2
import numpy as np
import mediapipe as mp
import time
import pyautogui
from collections import deque
import pygame


class AIVirtualKeyboard:
    def __init__(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize mediapipe hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Keyboard parameters
        self.keyboard_opacity = 0.35  # 35% opacity
        self.hover_color = (50, 205, 50)  # Light green for hover
        self.normal_color = (200, 200, 200)  # Gray for normal keys
        self.text_color = (0, 0, 0)  # Black Text

        # Keyboard layout
        self.keys = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'BACKSPACE'],
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '-', '='],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\''],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/'],
            ['SPACE', 'SHIFT', 'CAPS', 'ENTER'],
            ['PAGE↑', 'PAGE↓', 'PAGE←', 'PAGE→', 'ZOOM+', 'ZOOM-']
        ]

        # Special keys
        self.special_keys = {'BACKSPACE', 'SPACE', 'SHIFT', 'CAPS', 'ENTER',
                             'PAGE↑', 'PAGE↓', 'PAGE←', 'PAGE→', 'ZOOM+', 'ZOOM-'}

        # Keystroke parameters
        self.current_key = None
        self.last_key = None
        self.click_time = 0
        self.click_cooldown = 0.3  # seconds

        # FPS calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.fps_values = deque(maxlen=10)  # Store last 10 FPS values for smoothing

        # Text input
        self.typed_text = ""
        self.cursor_position = 0
        self.shift_active = False
        self.caps_active = False

        # Word suggestion system
        self.current_word = ""
        self.suggestions = []

        # Common English words for suggestions
        self.common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not", "on", "with", "he",
            "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
            "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about",
            "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than",
            "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
            "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give",
            "day", "most", "us", "python", "code", "programming", "computer", "keyboard", "virtual", "typing",
            "gesture", "hand", "finger", "mouse", "click", "cursor", "screen", "webcam", "camera", "detection",
            "tracking", "opencv", "mediapipe", "image", "video", "frame", "zoom", "page", "scroll", "navigation"
        ]

        # Auto-correction dictionary
        self.corrections = {
            "teh": "the", "adn": "and", "taht": "that", "wiht": "with", "fo": "of", "hte": "the", "ahve": "have",
            "waht": "what", "thier": "their", "im": "I'm", "dont": "don't", "cant": "can't", "wont": "won't",
            "isnt": "isn't", "didnt": "didn't", "couldnt": "couldn't", "shouldnt": "shouldn't",
            "wouldnt": "wouldn't", "hasnt": "hasn't", "havent": "haven't", "doesnt": "doesn't"
        }

        # Zoom level
        self.zoom_level = 1.0

        # Initialize pygame for sound effects
        pygame.init()
        try:
            self.key_sound = pygame.mixer.Sound('keyboard_click.wav')
            self.volume = 0.3
            pygame.mixer.Sound.set_volume(self.key_sound, self.volume)
        except:
            print("Sound file not found. Continuing without sound.")
            self.key_sound = None

    def calculate_fps(self):
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time) if (
                                                                          self.new_frame_time - self.prev_frame_time) > 0 else 0
        self.prev_frame_time = self.new_frame_time
        self.fps_values.append(fps)
        return sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0

    def draw_keyboard(self, img, hover_key=None):
        overlay = img.copy()
        key_height = 50
        spacing = 5
        start_y = self.height - (len(self.keys) * (key_height + spacing) + spacing)

        # Draw text input box
        text_box_height = 40
        cv2.rectangle(overlay, (10, start_y - text_box_height - 10),
                      (self.width - 10, start_y - 10), (50, 50, 50), -1)

        # Show typed text
        text_to_display = self.typed_text[:self.cursor_position] + "|" + self.typed_text[self.cursor_position:]
        cv2.putText(overlay, text_to_display, (20, start_y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw suggestion bar
        if self.suggestions:
            suggestion_y = start_y - text_box_height - 40
            suggestion_width = self.width // len(self.suggestions)
            for i, suggestion in enumerate(self.suggestions):
                x1 = i * suggestion_width
                x2 = (i + 1) * suggestion_width
                cv2.rectangle(overlay, (int(x1), int(suggestion_y)),
                              (int(x2), int(suggestion_y + 30)), (70, 70, 70), -1)
                cv2.putText(overlay, suggestion, (int(x1 + 10), int(suggestion_y + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw keyboard keys
        for row_idx, row in enumerate(self.keys):
            key_width = (self.width - (len(row) + 1) * spacing) // len(row)
            y = start_y + row_idx * (key_height + spacing)

            for key_idx, key in enumerate(row):
                # Adjust width for special keys
                current_key_width = key_width
                if key == 'SPACE':
                    current_key_width = key_width * 3
                elif key in {'BACKSPACE', 'ENTER', 'SHIFT', 'CAPS'}:
                    current_key_width = key_width * 1.5

                x = spacing + key_idx * (key_width + spacing)

                # Draw key background
                color = self.hover_color if key == hover_key else self.normal_color
                cv2.rectangle(overlay, (int(x), int(y)),
                              (int(x + current_key_width), int(y + key_height)), color, -1)

                # Draw key label
                display_key = key
                if key == 'SPACE':
                    display_key = '_____'

                # Adjust text position for key
                text_size = cv2.getTextSize(display_key, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = x + (current_key_width - text_size[0]) // 2
                text_y = y + (key_height + text_size[1]) // 2

                cv2.putText(overlay, display_key, (int(text_x), int(text_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

        # Apply overlay with opacity
        cv2.addWeighted(overlay, self.keyboard_opacity, img, 1 - self.keyboard_opacity, 0, img)
        return img

    def detect_key_hover(self, x, y):
        key_height = 50
        spacing = 5
        start_y = self.height - (len(self.keys) * (key_height + spacing) + spacing)

        # Check suggestion bar
        if self.suggestions:
            suggestion_y = start_y - 80
            suggestion_width = self.width // len(self.suggestions)
            if suggestion_y <= y <= suggestion_y + 30:
                suggestion_idx = int(x // suggestion_width)
                if 0 <= suggestion_idx < len(self.suggestions):
                    return f"SUGGESTION_{suggestion_idx}"

        # Check keys
        for row_idx, row in enumerate(self.keys):
            key_width = (self.width - (len(row) + 1) * spacing) // len(row)
            y_start = start_y + row_idx * (key_height + spacing)
            y_end = y_start + key_height

            if y_start <= y <= y_end:
                accumulated_width = spacing
                for key_idx, key in enumerate(row):
                    # Adjust width for special keys
                    current_key_width = key_width
                    if key == 'SPACE':
                        current_key_width = key_width * 3
                    elif key in {'BACKSPACE', 'ENTER', 'SHIFT', 'CAPS'}:
                        current_key_width = key_width * 1.5

                    x_start = accumulated_width
                    x_end = x_start + current_key_width

                    if x_start <= x <= x_end:
                        return key

                    accumulated_width += current_key_width + spacing

        return None

    def generate_suggestions(self, word):
        suggestions = []

        # Check for autocorrection first
        if word.lower() in self.corrections:
            suggestions.append(self.corrections[word.lower()])

        # Find words that start with the current input
        if word:
            count = 0
            for common_word in self.common_words:
                if common_word.startswith(word.lower()) and common_word != word.lower():
                    suggestions.append(common_word)
                    count += 1
                    if count >= 4:  # Limit to 4 suggestions
                        break

        return suggestions

    def handle_key_press(self, key):
        if key is None:
            return

        # Handle suggestion selection
        if isinstance(key, str) and key.startswith("SUGGESTION_"):
            suggestion_idx = int(key.split("_")[1])
            if 0 <= suggestion_idx < len(self.suggestions):
                words = self.typed_text[:self.cursor_position].split()
                if words:
                    # Replace the last word with the selected suggestion
                    typed_without_current = self.typed_text[:self.cursor_position - len(self.current_word)]
                    self.typed_text = (typed_without_current + self.suggestions[suggestion_idx] +
                                       self.typed_text[self.cursor_position:])
                    self.cursor_position = len(typed_without_current) + len(self.suggestions[suggestion_idx])
                    self.current_word = ""
                    self.suggestions = []
                return

        # Play key sound
        if self.key_sound:
            self.key_sound.play()

        # Handle special keys
        if key == 'BACKSPACE':
            if self.cursor_position > 0:
                self.typed_text = (self.typed_text[:self.cursor_position - 1] +
                                   self.typed_text[self.cursor_position:])
                self.cursor_position -= 1
                # Update current word and suggestions
                words = self.typed_text[:self.cursor_position].split()
                self.current_word = words[-1] if words else ""
                self.suggestions = self.generate_suggestions(self.current_word)
        elif key == 'SPACE':
            # Auto-correct when space is pressed
            words = self.typed_text[:self.cursor_position].split()
            if words and words[-1].lower() in self.corrections:
                corrected = self.corrections[words[-1].lower()]
                # Replace the last word with the corrected version
                typed_without_current = self.typed_text[:self.cursor_position - len(words[-1])]
                self.typed_text = typed_without_current + corrected + " " + self.typed_text[self.cursor_position:]
                self.cursor_position = len(typed_without_current) + len(corrected) + 1
            else:
                self.typed_text = (self.typed_text[:self.cursor_position] + " " +
                                   self.typed_text[self.cursor_position:])
                self.cursor_position += 1

            self.current_word = ""
            self.suggestions = []
        elif key == 'ENTER':
            self.typed_text = (self.typed_text[:self.cursor_position] + "\n" +
                               self.typed_text[self.cursor_position:])
            self.cursor_position += 1
            self.current_word = ""
            self.suggestions = []
        elif key == 'SHIFT':
            self.shift_active = not self.shift_active
        elif key == 'CAPS':
            self.caps_active = not self.caps_active
        elif key == 'PAGE↑':
            pyautogui.press('pageup')
        elif key == 'PAGE↓':
            pyautogui.press('pagedown')
        elif key == 'PAGE←':
            pyautogui.press('home')
        elif key == 'PAGE→':
            pyautogui.press('end')
        elif key == 'ZOOM+':
            self.zoom_level = min(2.0, self.zoom_level + 0.1)
        elif key == 'ZOOM-':
            self.zoom_level = max(0.5, self.zoom_level - 0.1)
        # Regular character keys
        elif len(key) == 1:
            char = key.upper() if (self.shift_active ^ self.caps_active) else key.lower()
            self.typed_text = (self.typed_text[:self.cursor_position] + char +
                               self.typed_text[self.cursor_position:])
            self.cursor_position += 1

            # Update current word and get suggestions
            words = self.typed_text[:self.cursor_position].split()
            self.current_word = words[-1] if words else ""
            self.suggestions = self.generate_suggestions(self.current_word)

    def detect_gestures(self, hand_landmarks):
        # Get index and middle finger tip positions
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        # Convert to pixel coordinates
        index_x, index_y = int(index_tip.x * self.width), int(index_tip.y * self.height)
        middle_x, middle_y = int(middle_tip.x * self.width), int(middle_tip.y * self.height)

        # Calculate distance between fingers
        finger_distance = np.sqrt((index_x - middle_x) ** 2 + (index_y - middle_y) ** 2)

        # Detect cursor position - use the middle point between index and middle finger
        cursor_x = (index_x + middle_x) // 2
        cursor_y = (index_y + middle_y) // 2

        # Increase the threshold for click detection
        is_click = finger_distance < 40  # Increased to 40 for better detection

        return cursor_x, cursor_y, is_click, finger_distance

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Flip frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)

            # Apply zoom if needed
            if self.zoom_level != 1.0:
                h, w = frame.shape[:2]
                # Calculate new dimensions
                new_h, new_w = int(h * self.zoom_level), int(w * self.zoom_level)
                # Calculate crop dimensions
                crop_h, crop_w = min(h, new_h), min(w, new_w)
                # Calculate offsets
                offset_h, offset_w = (new_h - h) // 2, (new_w - w) // 2

                # Create a larger canvas
                zoomed_frame = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                # Place the original frame in the center
                zoomed_frame[max(0, offset_h):max(0, offset_h) + crop_h,
                max(0, offset_w):max(0, offset_w) + crop_w] = \
                    frame[max(0, -offset_h):max(0, -offset_h) + crop_h,
                    max(0, -offset_w):max(0, -offset_w) + crop_w]

                # Resize back to original dimensions
                frame = cv2.resize(zoomed_frame, (w, h))

            # Convert to RGB for mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame for hand detection
            results = self.hands.process(rgb_frame)

            # Detect which key is being hovered
            hover_key = None
            finger_distance = 0  # Initialize finger_distance

            # If hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Detect gestures
                    cursor_x, cursor_y, is_click, finger_distance = self.detect_gestures(hand_landmarks)

                    # Add distance debugging info
                    cv2.putText(frame, f"Dist: {finger_distance:.1f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Draw cursor
                    cv2.circle(frame, (int(cursor_x), int(cursor_y)), 10, (0, 255, 255), -1)

                    # Detect key hover
                    hover_key = self.detect_key_hover(cursor_x, cursor_y)

                    # Handle key press if click gesture detected
                    current_time = time.time()
                    if is_click and (current_time - self.click_time > self.click_cooldown):
                        cv2.circle(frame, (int(cursor_x), int(cursor_y)), 15, (0, 0, 255), 2)
                        if hover_key != self.last_key:  # Prevent multiple clicks on same key
                            self.handle_key_press(hover_key)
                            self.last_key = hover_key
                            self.click_time = current_time
                    elif not is_click:
                        self.last_key = None

            # Draw keyboard
            frame = self.draw_keyboard(frame, hover_key)

            # Calculate and display FPS
            fps = self.calculate_fps()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('AI Virtual Keyboard', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    keyboard = AIVirtualKeyboard()
    keyboard.run()
