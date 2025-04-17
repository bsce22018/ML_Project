from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
from pynput.keyboard import Key, Controller
import time

app = Flask(__name__)
CORS(app, resources={r"/gesture": {"origins": "*"}})

# MediaPipe setup
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
fingerTipIds = [4, 8, 12, 16, 20]

# Keyboard control
keyboard = Controller()
last_action = None
last_time = time.time()

def control_keyboard(gesture):
    global last_action, last_time

    # Cooldown to prevent spamming
    now = time.time()
    if now - last_time < 0.3:
        return
    last_time = now

    if gesture == "GAS" and last_action != "GAS":
        keyboard.press(Key.right)
        keyboard.release(Key.left)
        last_action = "GAS"

    elif gesture == "BRAKE" and last_action != "BRAKE":
        keyboard.press(Key.left)
        keyboard.release(Key.right)
        last_action = "BRAKE"

    elif gesture == "NONE":
        keyboard.release(Key.right)
        keyboard.release(Key.left)
        last_action = "NONE"

def detect_gesture(image_np):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    landmarks_list = []

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        for index, lm in enumerate(hand_landmarks.landmark):
            h, w, _ = image_np.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks_list.append((index, cx, cy))

        fingers_open = []
        for tipId in fingerTipIds:
            if tipId == 4:
                fingers_open.append(1 if landmarks_list[tipId][1] > landmarks_list[tipId - 1][1] else 0)
            else:
                fingers_open.append(1 if landmarks_list[tipId][2] < landmarks_list[tipId - 2][2] else 0)

        count = fingers_open.count(1)
        if count == 5:
            return "GAS"
        elif count == 0:
            return "BRAKE"

    return "NONE"

@app.route("/gesture", methods=["POST"])
def gesture():
    data = request.json.get("image")
    if not data:
        return jsonify({"error": "No image provided"}), 400

    try:
        base64_data = data.split(",")[1]
        image_data = base64.b64decode(base64_data)
        np_arr = np.frombuffer(image_data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_np is None:
            return jsonify({"error": "Invalid image data"}), 400
    except Exception as e:
        return jsonify({"error": f"Error decoding image: {str(e)}"}), 400

    gesture_result = detect_gesture(image_np)
    control_keyboard(gesture_result)  # This is the new action
    return jsonify({"gesture": gesture_result})

if __name__ == "__main__":
    app.run(debug=True)
