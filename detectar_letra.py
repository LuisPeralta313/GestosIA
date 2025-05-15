import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Config
MODEL_PATH = "modelo_asl.h5"
IMG_SIZE = 128
CLASSES_PATH = r"E:\3PrimerCiclo2025\IA\ProyectoFinal-ia\asl_alphabet_train"

# Cargar modelo
model = load_model(MODEL_PATH)
clases = sorted(os.listdir(CLASSES_PATH))
clases_dict = {i: clase for i, clase in enumerate(clases)}

# Cámara
cap = cv2.VideoCapture(0)
print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Región de interés
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocesamiento
    image = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predicción
    predictions = model.predict(image)
    confidence = np.max(predictions[0])
    pred_index = np.argmax(predictions[0])
    predicted_letter = clases_dict[pred_index]

    # Mostrar si es confiable
    if confidence > 0.7:
        cv2.putText(frame, f"{predicted_letter} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Detector ASL", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
