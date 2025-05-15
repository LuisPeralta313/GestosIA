import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

MODEL_PATH = "modelo_asl.h5"
TEST_FOLDER = r"E:\3PrimerCiclo2025\IA\ProyectoFinal-ia\asl_alphabet_test"
TRAIN_FOLDER = r"E:\3PrimerCiclo2025\IA\ProyectoFinal-ia\asl_alphabet_train"
IMG_SIZE = 128

# Cargar modelo y clases
model = load_model(MODEL_PATH)
clases = sorted(os.listdir(TRAIN_FOLDER))
clases_dict = {i: clase for i, clase in enumerate(clases))

# Probar test
for filename in os.listdir(TEST_FOLDER):
    if filename.endswith(".jpg"):
        path = os.path.join(TEST_FOLDER, filename)
        label_real = filename.split("_")[0]

        image = cv2.imread(path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        pred = model.predict(image)
        pred_index = np.argmax(pred[0])
        pred_label = clases_dict[pred_index]
        confidence = np.max(pred[0])

        print(f"{filename}: {label_real} → {pred_label} ({confidence:.2f}) {'✅' if label_real == pred_label else '❌'}")
