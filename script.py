import cv2
import mediapipe as mp
import os
import csv

# Config
SIGNS = ["two","five"]  # जितने sign चाहिए
DATASET_PATH = "dataset_landmarks"
os.makedirs(DATASET_PATH, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

for SIGN_NAME in SIGNS:
    csv_file_path = os.path.join(DATASET_PATH, f"{SIGN_NAME}.csv")
    csv_file = open(csv_file_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    print(f"Collecting data for: {SIGN_NAME}")
    print("Press 'q' to move to next sign")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror effect
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                row = [SIGN_NAME]  # label
                for lm in handLms.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                csv_writer.writerow(row)

                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow(f"Collecting {SIGN_NAME}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    csv_file.close()

cap.release()
cv2.destroyAllWindows()
print("✅ Dataset collection complete")
