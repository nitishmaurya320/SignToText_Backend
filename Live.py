import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Sentence buffer
sentence = ""
predictions_queue = deque(maxlen=15)  # last 15 predictions store

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Extract landmarks
            data = []
            for lm in handLms.landmark:
                data.extend([lm.x, lm.y, lm.z])

            data = np.array(data).reshape(1, -1)
            prediction = model.predict(data)[0]

            # Store last predictions
            predictions_queue.append(prediction)

            # Majority voting (smooth prediction)
            if len(predictions_queue) == predictions_queue.maxlen:
                most_common = max(set(predictions_queue), key=predictions_queue.count)

                # Add word to sentence if different from last word
                if not sentence.endswith(most_common + " "):
                    sentence += most_common + " "

            # Show prediction
            cv2.putText(frame, prediction, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Show sentence
    cv2.putText(frame, "Sentence: " + sentence, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Live Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
