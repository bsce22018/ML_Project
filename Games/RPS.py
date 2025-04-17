from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import random
import mediapipe as mp

app = Flask(__name__)
CORS(app)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
choices = ['Rock', 'Paper', 'Scissors']
scores = {'player': 0, 'ai': 0, 'ties': 0}

def classify_gesture(finger_states):
    if finger_states == [0, 0, 0, 0, 0]:
        return "Rock"
    elif finger_states == [1, 1, 1, 1, 1]:
        return "Paper"
    elif finger_states == [0, 1, 1, 0, 0]:
        return "Scissors"
    return "Unknown"

def get_finger_states(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    states = []
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
        states.append(1)
    else:
        states.append(0)
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            states.append(1)
        else:
            states.append(0)
    return states

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = base64.b64decode(data['frame'].split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)

    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    current_gesture = "Unknown"
    landmarks_list = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            finger_states = get_finger_states(hand_landmarks)
            current_gesture = classify_gesture(finger_states)
            for lm in hand_landmarks.landmark:
                landmarks_list.append({'x': lm.x, 'y': lm.y, 'z': lm.z})

    ai_move = random.choice(choices)
    result_text = "Unknown"

    if current_gesture != "Unknown":
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

    return jsonify({
        "player_move": current_gesture,
        "ai_move": ai_move,
        "result": result_text,
        "scores": scores,
        "landmarks": landmarks_list  # Send back hand landmarks
    })

@app.route('/reset', methods=['POST'])
def reset():
    global scores
    scores = {'player': 0, 'ai': 0, 'ties': 0}
    return jsonify({"status": "reset", "scores": scores})

if __name__ == '__main__':
    app.run(debug=True)
