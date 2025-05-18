import pickle
import cv2
import mediapipe as mp
import numpy as np

# Cargar modelo
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Captura de video
cap = cv2.VideoCapture(0)

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Texto acumulado
accumulated_text = ""
last_prediction = ""

# Limpiar archivo de texto al iniciar
with open("resultado.txt", "w") as f:
    f.write("")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    predicted_character = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = prediction[0]

        # Solo guardar si cambia la predicción
        if predicted_character != last_prediction:
            accumulated_text += predicted_character
            with open("resultado.txt", "a") as f:
                f.write(predicted_character)
            last_prediction = predicted_character

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # Mostrar texto acumulado con letra más pequeña y color verde claro
    cv2.putText(frame, accumulated_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (144, 238, 144), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Detector', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
        accumulated_text += " "
        with open("resultado.txt", "a") as f:
            f.write(" ")
    elif key == ord('c'):
        accumulated_text = ""
        with open("resultado.txt", "a") as f:
            f.write("\n[Texto limpio]\n")

cap.release()
cv2.destroyAllWindows()
