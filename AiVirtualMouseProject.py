import cv2
import numpy as np
import time
import autopy
import pyautogui
import math
import mediapipe as mp


class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.maxHands = max_num_hands
        self.detectionCon = min_detection_confidence
        self.trackCon = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []

        if self.results.multi_hand_landmarks:
            try:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy])
                    if draw and id in self.tipIds:
                        cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

                if xList and yList:
                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)
                    bbox = xmin, ymin, xmax, ymax

                    if draw:
                        cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                                      (0, 255, 0), 2)
            except:
                pass

        return self.lmList, bbox

    def fingersUp(self):
        self.fingers = []
        if len(self.lmList) >= 21:
            # Thumb
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                self.fingers.append(1)
            else:
                self.fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    self.fingers.append(1)
                else:
                    self.fingers.append(0)
        return self.fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    # Initialize variables
    wCam, hCam = 640, 480
    frameR = 100  # Frame reduction
    smoothening = 7

    # Previous time for FPS calculation
    pTime = 0

    # Previous location for smoothing
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    # Gesture state tracking
    current_mode = "None"
    prev_fingers_up = 0
    drag_mode = False
    drag_active = False  # Track if drag is currently active
    prev_drag_state = False
    scroll_continuous = False
    last_click_time = 0
    click_cooldown = 0.3  # Seconds between clicks
    gesture_change_time = 0
    gesture_cooldown = 0.5  # Cooldown for gesture changes
    gesture_start_y = 0
    gesture_active = False
    scroll_speed = 5

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    # Initialize hand detector
    detector = HandDetector(max_num_hands=1)

    # Get screen size
    wScr, hScr = autopy.screen.size()

    while True:
        try:
            # 1. Find hand landmarks
            success, img = cap.read()
            if not success:
                print("Failed to grab frame")
                continue

            img = detector.findHands(img)
            lmList, bbox = detector.findPosition(img)

            # 2. Get finger tip positions and check which fingers are up
            if len(lmList) != 0:
                fingers = detector.fingersUp()
                fingers_up_count = sum(fingers)
                cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

                # Default position for index finger tip
                x1, y1 = lmList[8][1:]

                # Get current time
                current_time = time.time()

                # State transitions with cooldown to prevent rapid toggling
                if current_time - gesture_change_time > gesture_cooldown:
                    if prev_fingers_up != fingers_up_count:
                        gesture_change_time = current_time
                        gesture_start_y = y1  # Reset gesture start position
                        gesture_active = True
                        current_mode = "Transitioning"

                # 3. MODE DETECTION AND ACTIONS
                # ------MODE 1: Two fingers up - Mouse Movement------
                if fingers_up_count == 2 and fingers[1] == 1 and fingers[2] == 1:
                    current_mode = "Move"

                    # a. Convert coordinates to screen position
                    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                    # b. Smooth values
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening

                    # c. Move mouse
                    try:
                        autopy.mouse.move(wScr - clocX, clocY)
                        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        plocX, plocY = clocX, clocY
                    except Exception as e:
                        print(f"Mouse movement error: {e}")

                # ------MODE 2: Index finger down - Left Click------
                elif fingers_up_count == 1 and fingers[1] == 0 and fingers[2] == 1:
                    current_mode = "Left Click"
                    # Left click with cooldown
                    if current_time - last_click_time > click_cooldown:
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                        pyautogui.click(button='left')
                        last_click_time = current_time

                # ------MODE 3: Middle finger down - Right Click------
                elif fingers_up_count == 1 and fingers[1] == 1 and fingers[2] == 0:
                    current_mode = "Right Click"
                    # Right click with cooldown
                    if current_time - last_click_time > click_cooldown:
                        cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
                        pyautogui.click(button='right')
                        last_click_time = current_time

                # ------MODE 4: Zero fingers up (Closed Fist) - Start Drag------
                elif all(finger == 0 for finger in fingers):
                    # If not already in drag mode, start it
                    if not drag_active:
                        current_mode = "Start Drag"
                        if current_time - last_click_time > click_cooldown:
                            cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)
                            # Press and hold mouse button to start drag
                            pyautogui.mouseDown()
                            drag_active = True
                            last_click_time = current_time
                    else:
                        # Continue dragging
                        current_mode = "Dragging"
                        # Convert coordinates
                        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                        # Smooth values
                        clocX = plocX + (x3 - plocX) / smoothening
                        clocY = plocY + (y3 - plocY) / smoothening

                        # Move mouse while dragging (don't press again, just move)
                        try:
                            autopy.mouse.move(wScr - clocX, clocY)
                            cv2.circle(img, (x1, y1), 15, (0, 255, 255), cv2.FILLED)
                            plocX, plocY = clocX, clocY
                        except Exception as e:
                            print(f"Drag movement error: {e}")

                # If fingers were previously closed but now open, end the drag
                elif drag_active and fingers_up_count > 0:
                    current_mode = "End Drag"
                    pyautogui.mouseUp()  # Release mouse button
                    drag_active = False
                    last_click_time = current_time

                # ------MODE 6: Three fingers up - Single Scroll------
                elif fingers_up_count == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    current_mode = "Single Scroll"
                    # Calculate vertical movement
                    y_move = gesture_start_y - y1

                    # Scroll based on movement
                    if abs(y_move) > 20:  # Movement threshold
                        if y_move > 0:  # Moving up
                            cv2.putText(img, "SCROLL UP", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                            pyautogui.scroll(100)  # Positive value scrolls up
                        else:  # Moving down
                            cv2.putText(img, "SCROLL DOWN", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                            pyautogui.scroll(-100)  # Negative value scrolls down

                        # Reset gesture start position
                        gesture_start_y = y1

                # ------MODE 7: Four fingers up - Continuous Scroll------
                elif fingers_up_count == 4:
                    current_mode = "Continuous Scroll"
                    # Calculate vertical movement
                    y_move = gesture_start_y - y1

                    # Start continuous scrolling
                    scroll_continuous = True

                    # Adjust scroll speed based on movement
                    if abs(y_move) > 10:
                        scroll_speed = max(1, min(15, abs(y_move) // 10))  # Scale between 1-15

                        if y_move > 0:  # Moving up
                            cv2.putText(img, f"CONTINUOUS UP (Speed: {scroll_speed})", (50, 100),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                            pyautogui.scroll(scroll_speed * 10)  # Scale with speed
                        else:  # Moving down
                            cv2.putText(img, f"CONTINUOUS DOWN (Speed: {scroll_speed})", (50, 100),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                            pyautogui.scroll(-scroll_speed * 10)  # Scale with speed
                else:
                    scroll_continuous = False

                # ------MODE 8: Five fingers up - Zoom------
                if fingers_up_count == 5:
                    current_mode = "Zoom"
                    # Calculate vertical movement
                    y_move = gesture_start_y - y1

                    # Zoom based on movement with cooldown
                    if abs(y_move) > 20 and current_time - last_click_time > click_cooldown:
                        if y_move > 0:  # Moving up - zoom in
                            cv2.putText(img, "ZOOM IN", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                            pyautogui.hotkey('ctrl', '+')
                        else:  # Moving down - zoom out
                            cv2.putText(img, "ZOOM OUT", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                            pyautogui.hotkey('ctrl', '-')

                        # Reset cooldown and gesture start position
                        last_click_time = current_time
                        gesture_start_y = y1

                # Store previous state
                prev_fingers_up = fingers_up_count
                prev_drag_state = all(finger == 0 for finger in fingers)
            else:
                # If hand is no longer detected and drag is active, end the drag
                if drag_active:
                    pyautogui.mouseUp()
                    drag_active = False

                # Reset modes when no hand detected
                drag_mode = False
                scroll_continuous = False

            # 4. Display current mode and drag status
            cv2.putText(img, f"Mode: {current_mode}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
            if drag_active:
                cv2.putText(img, "DRAGGING ACTIVE", (10, 110), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

            # 5. Calculate and display FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

            # 6. Display the image
            cv2.imshow("Advanced Virtual Mouse", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                # Ensure drag is released before quitting
                if drag_active:
                    pyautogui.mouseUp()
                break

        except Exception as e:
            print(f"Error: {e}")
            # Ensure drag is released on error
            if 'drag_active' in locals() and drag_active:
                pyautogui.mouseUp()
                drag_active = False
            continue

    # 7. Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
