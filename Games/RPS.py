from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import mediapipe as mp
import pickle

app = Flask(__name__)
CORS(app)

# Load trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

choices = ['Rock', 'Paper', 'Scissors']
scores = {'player': 0, 'ai': 0, 'ties': 0}

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
        hand_landmarks = result.multi_hand_landmarks[0]
        features = []
        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
            landmarks_list.append({'x': lm.x, 'y': lm.y, 'z': lm.z})

        current_gesture = model.predict([features])[0]

    ai_move = np.random.choice(choices)
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
        "landmarks": landmarks_list
    })

@app.route('/reset', methods=['POST'])
def reset():
    global scores
    scores = {'player': 0, 'ai': 0, 'ties': 0}
    return jsonify({"status": "reset", "scores": scores})

if __name__ == '__main__':
    app.run(debug=True)
