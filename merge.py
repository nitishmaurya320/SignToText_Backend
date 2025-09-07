import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# ---- Dataset Load ----
dataset_path = "dataset_landmarks"
files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".csv")]

dataframes = [pd.read_csv(f, header=None) for f in files]
df = pd.concat(dataframes, ignore_index=True)

# Labels और Features
labels = df.iloc[:, 0]
features = df.iloc[:, 1:]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# ---- Model Train ----
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Accuracy check
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---- Save Model ----
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as model.pkl")
