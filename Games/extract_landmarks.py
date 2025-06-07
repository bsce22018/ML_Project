import cv2
import mediapipe as mp
import os
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
dataset = []

base_dir = "dataset"  # Folder with Rock, Paper, Scissors

for label in os.listdir(base_dir):
    label_path = os.path.join(base_dir, label)
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(label_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            row.append(label)
            dataset.append(row)

# Save to CSV
columns = [f'{coord}{i}' for i in range(21) for coord in ('x', 'y', 'z')] + ['label']
df = pd.DataFrame(dataset, columns=columns)
df.to_csv('gesture_data.csv', index=False)
print(f"[INFO] Saved {len(df)} samples to gesture_data.csv")
