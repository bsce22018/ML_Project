import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv("gesture_data.csv")
X = df.drop("label", axis=1)
y = df["label"]

# Step 1: Split into train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# Step 2: Split temp into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Train model on training set
model = RandomForestClassifier()
model.fit(X_train, y_train)

# (Optional) Evaluate on validation set
print("[INFO] Validation Results:")
y_val_pred = model.predict(X_val)
print(classification_report(y_val, y_val_pred))

# (Optional) Evaluate on test set
print("[INFO] Test Results:")
y_test_pred = model.predict(X_test)
print(classification_report(y_test, y_test_pred))

# Save the model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("[INFO] Model trained and saved as gesture_model.pkl")
