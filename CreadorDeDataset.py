import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Configurar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
expected_landmarks = 21  # Número de landmarks por mano
expected_features = expected_landmarks * 2  # 42 valores (x, y por landmark)

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # Ignora archivos que no son carpetas

    for img_path in os.listdir(dir_path):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            print(f"Advertencia: No se pudo leer la imagen {os.path.join(dir_path, img_path)}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:  # Solo una mano
            hand_landmarks = results.multi_hand_landmarks[0]  # Tomar la primera (y única) mano
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            if len(data_aux) == expected_features:  # Verificar que el vector tenga 42 valores
                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"Advertencia: Vector de características inválido para {os.path.join(dir_path, img_path)}")
        else:
            print(f"Advertencia: No se detectó exactamente una mano en {os.path.join(dir_path, img_path)}")

# Convertir a arreglos de NumPy y verificar formas
data = np.asarray(data)
labels = np.asarray(labels)

print(f"Datos generados: {len(data)} muestras")
print(f"Forma de los datos: {data.shape}")
print(f"Forma de las etiquetas: {labels.shape}")

# Guardar datos
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

# Cerrar MediaPipe Hands
hands.close()