import cv2
import numpy as np
import sys
import os
from tensorflow.keras.models import load_model


# ============================================================
# 1. MODEL PATH
# ============================================================

MODEL_PATH = "ML_Model/emotion_detection_model.h5"


# ============================================================
# 2. EMOTION LABELS
# ============================================================

emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}


# ============================================================
# 3. LOAD EMOTION MODEL
# ============================================================

print("Loading emotion detection model...")

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found:")
    print(MODEL_PATH)
    sys.exit(1)

try:
    model = load_model(MODEL_PATH)
    print("Emotion model loaded successfully")

except Exception as e:
    print("ERROR: Failed to load emotion model")
    print(e)
    sys.exit(1)


# ============================================================
# 4. LOAD HAAR CASCADE
# ============================================================

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("ERROR: Haar Cascade could not be loaded")
    sys.exit(1)

print("Face detector loaded successfully")


# ============================================================
# 5. PREPROCESS FACE
# ============================================================

def preprocess_face(face_img):

    # Convert BGR image to grayscale
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Resize to model input size
    gray_face = cv2.resize(gray_face, (48, 48))

    # Normalize pixel values
    gray_face = gray_face.astype("float32") / 255.0

    # Reshape:
    # (48, 48) -> (1, 48, 48, 1)
    gray_face = np.expand_dims(gray_face, axis=0)
    gray_face = np.expand_dims(gray_face, axis=-1)

    return gray_face


# ============================================================
# 6. OPEN WEBCAM - MACOS
# ============================================================

print("Opening webcam...")

# CAP_AVFOUNDATION is the correct backend for macOS
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# If AVFoundation fails, try the default backend
if not cap.isOpened():

    print("AVFoundation camera failed.")
    print("Trying default camera backend...")

    cap.release()
    cap = cv2.VideoCapture(0)


# Check whether camera opened successfully
if not cap.isOpened():

    print()
    print("==========================================")
    print("ERROR: Webcam not accessible")
    print("==========================================")
    print()
    print("Please check:")
    print("1. macOS Camera permission")
    print("2. Camera is not being used by another app")
    print("3. Restart Terminal/VS Code")
    print()
    sys.exit(1)


# ============================================================
# 7. CAMERA SETTINGS
# ============================================================

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Webcam opened successfully")
print()
print("==========================================")
print("Emotion Detection Started")
print("Press 'q' to quit")
print("==========================================")


# ============================================================
# 8. MAIN EMOTION DETECTION LOOP
# ============================================================

while True:

    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:

        print("ERROR: Failed to grab frame")
        break


    # --------------------------------------------------------
    # Flip frame horizontally
    # Makes webcam behave like a mirror
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # Convert frame to grayscale
    # --------------------------------------------------------

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )


    # --------------------------------------------------------
    # Process every detected face
    # --------------------------------------------------------

    for (x, y, w, h) in faces:

        # Make sure coordinates are inside the frame
        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)


        # Extract face
        face_roi = frame[y1:y2, x1:x2]


        # Skip empty face
        if face_roi.size == 0:
            continue


        try:

            # ------------------------------------------------
            # Preprocess face
            # ------------------------------------------------

            processed_face = preprocess_face(face_roi)


            # ------------------------------------------------
            # Predict emotion
            # ------------------------------------------------

            predictions = model.predict(
                processed_face,
                verbose=0
            )


            # ------------------------------------------------
            # Find highest probability
            # ------------------------------------------------

            emotion_index = int(np.argmax(predictions))

            confidence = float(
                np.max(predictions) * 100
            )


            # ------------------------------------------------
            # Get emotion name
            # ------------------------------------------------

            emotion_text = emotion_labels.get(
                emotion_index,
                "Unknown"
            )


        except Exception as e:

            print("Prediction error:", e)
            continue


        # ====================================================
        # DRAW FACE RECTANGLE
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ====================================================
        # CREATE EMOTION LABEL
        # ====================================================

        label = f"{emotion_text} ({confidence:.1f}%)"


        # ====================================================
        # DRAW LABEL BACKGROUND
        # ====================================================

        label_y = y1 - 10

        # If label goes above screen, put it below face
        if label_y < 30:
            label_y = y2 + 30


        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )


        # Draw filled rectangle behind text
        cv2.rectangle(
            frame,
            (x1, label_y - text_height - baseline - 5),
            (x1 + text_width + 5, label_y + 5),
            (0, 0, 0),
            -1
        )


        # ====================================================
        # DRAW EMOTION TEXT
        # ====================================================

        cv2.putText(
            frame,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # DISPLAY FRAME
    # ========================================================

    cv2.imshow(
        "Emotion Detection System",
        frame
    )


    # ========================================================
    # QUIT WHEN 'q' IS PRESSED
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# 9. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("==========================================")
print("Program terminated successfully")
print("==========================================")