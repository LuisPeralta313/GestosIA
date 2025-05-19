import os
import cv2
import string

# Carpeta donde se guardarán las imágenes
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Lista de letras del alfabeto
classes = list(string.ascii_uppercase)  # ['A', 'B', ..., 'Z']
dataset_size = 250  # Cantidad de imágenes por letra

cap = cv2.VideoCapture(0)

for label in classes:
    class_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print(f'📸 Preparando para capturar la letra: {label}')
    print('👉 Mostrá la seña y presioná "q" para comenzar')

    # Espera a que el usuario presione "q" para empezar
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Letra: {label} - Presiona "q" para capturar',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Captura automática de imágenes
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        img_path = os.path.join(class_path, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

print('✅ Captura finalizada.')
cap.release()
cv2.destroyAllWindows()
