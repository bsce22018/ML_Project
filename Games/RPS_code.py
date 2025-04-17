import cv2
import mediapipe as mp
import random
import time
import winsound
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Open camera
cap = cv2.VideoCapture(0)

# Reduce resolution (for faster processing)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

choices = ['Rock', 'Paper', 'Scissors']
scores = {'player': 0, 'ai': 0, 'ties': 0}
game_state = 'waiting'  # waiting, countdown, playing
countdown_start = 0
last_gesture = None
gesture_confidence = 0
required_confidence = 5  # Number of consistent detections needed

def play_sound(result):
    if result == "You win!":
        winsound.Beep(1000, 500)  # Higher pitch for win
    elif result == "AI wins!":
        winsound.Beep(500, 500)   # Lower pitch for loss
    else:
        winsound.Beep(750, 300)   # Medium pitch for tie

def classify_gesture(finger_states):
    if finger_states == [0, 0, 0, 0, 0]:
        return "Rock"
    elif finger_states == [1, 1, 1, 1, 1]:
        return "Paper"
    elif finger_states == [0, 1, 1, 0, 0]:
        return "Scissors"
    else:
        return "Unknown"

def get_finger_states(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    thumb_tip = 4
    states = []

    # Thumb state
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
        states.append(1)
    else:
        states.append(0)

    # Finger states
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            states.append(1)
        else:
            states.append(0)

    return states

def draw_score(frame):
    cv2.putText(frame, f"Player: {scores['player']}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"AI: {scores['ai']}", (540, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Ties: {scores['ties']}", (270, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_gesture_guide(frame):
    guide_text = "Rock: Closed fist | Paper: Open hand | Scissors: Victory sign"
    cv2.putText(frame, guide_text, (10, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def reset_game():
    global scores, game_state, countdown_start, last_gesture, gesture_confidence, frame_counter
    scores = {'player': 0, 'ai': 0, 'ties': 0}
    game_state = 'waiting'
    countdown_start = 0
    last_gesture = None
    gesture_confidence = 0
    frame_counter = 0  # Reset frame counter

frame_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    frame_counter += 1
    if frame_counter % 2 == 0:  # Process every other frame to reduce lag
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Draw score and guide
        draw_score(frame)
        draw_gesture_guide(frame)

        current_gesture = "Unknown"
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                finger_states = get_finger_states(hand_landmarks)
                current_gesture = classify_gesture(finger_states)

        # Gesture confidence tracking
        if current_gesture == last_gesture and current_gesture != "Unknown":
            gesture_confidence += 1
        else:
            gesture_confidence = 0
        last_gesture = current_gesture

        # Game state machine
        if game_state == 'waiting':
            if gesture_confidence >= required_confidence:
                game_state = 'countdown'
                countdown_start = time.time()
            cv2.putText(frame, "Show your gesture!", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)

        elif game_state == 'countdown':
            elapsed = time.time() - countdown_start
            if elapsed < 3:
                countdown = 3 - int(elapsed)
                cv2.putText(frame, str(countdown), (w//2 - 40, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
            else:
                game_state = 'playing'
                ai_move = random.choice(choices)
                result_text = ""

                if current_gesture == ai_move:
                    result_text = "It's a tie!"
                    scores['ties'] += 1
                elif (current_gesture == "Rock" and ai_move == "Scissors") or \
                     (current_gesture == "Paper" and ai_move == "Rock") or \
                     (current_gesture == "Scissors" and ai_move == "Paper"):
                    result_text = "You win!"
                    scores['player'] += 1
                else:
                    result_text = "AI wins!"
                    scores['ai'] += 1

                play_sound(result_text)

                # Display the AI's response and result
                cv2.putText(frame, f"Your Move: {current_gesture}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"AI's Move: {ai_move}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
                cv2.putText(frame, result_text, (10, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 50, 50), 3)

                # Show the frame with the AI's response
                cv2.imshow("Rock Paper Scissors", frame)

                # Add a delay to allow the user to see the result
                cv2.waitKey(2000)  # Wait for 2 seconds (2000 milliseconds)

                # Reset the game state
                game_state = 'waiting'
                gesture_confidence = 0

        # Add reset option
        cv2.putText(frame, "Press 'R' to reset", (10, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.imshow("Rock Paper Scissors", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('r'):  # Reset game
        reset_game()

cap.release()
cv2.destroyAllWindows()

