import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Cargar modelo
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Texto acumulado
accumulated_text = ""
last_prediction = ""
consecutive_count = 0
# Número de frames consecutivos requeridos. Cambiar si se cree que es muy rápido o lento.
STABILITY_THRESHOLD = 8

# Limpiar archivo de texto al iniciar
with open("resultado.txt", "w") as f:
    f.write("")

# Crear ventana principal
root = tk.Tk()
root.title("Detector de Lenguaje de Señas")

label_text = tk.Label(root, text="", font=("Helvetica", 16), fg="green")
label_text.pack()

canvas = tk.Label(root)
canvas.pack()

# Botones
frame_buttons = tk.Frame(root)
frame_buttons.pack()

def add_space():
    global accumulated_text
    accumulated_text += " "
    with open("resultado.txt", "a") as f:
        f.write(" ")

def clear_text():
    global accumulated_text
    if accumulated_text:  # Verificar que haya texto para eliminar
        accumulated_text = accumulated_text[:-1]  # Eliminar el último carácter
        # Reescribir el archivo con el texto actualizado
        with open("resultado.txt", "r") as f:
            current_content = f.read()
        if current_content:
            with open("resultado.txt", "w") as f:
                f.write(current_content[:-1])  # Eliminar el último carácter del archivo
        label_text.config(text=accumulated_text)  # Actualizar la interfaz

def exit_app():
    root.quit()

btn_space = tk.Button(frame_buttons, text="Espacio", command=add_space)
btn_space.grid(row=0, column=0, padx=10)

btn_clear = tk.Button(frame_buttons, text="Limpiar", command=clear_text)
btn_clear.grid(row=0, column=1, padx=10)

btn_exit = tk.Button(frame_buttons, text="Salir", command=exit_app)
btn_exit.grid(row=0, column=2, padx=10)

cap = cv2.VideoCapture(0)

current_stable_prediction = ""

def update_frame():
    global accumulated_text, last_prediction, consecutive_count, current_stable_prediction

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape

    results = hands.process(frame_rgb)
    predicted_character = ""

    if results.multi_hand_landmarks:
        x_ = []
        y_ = []
        data_aux = []

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

        # Letras sobre la Mano
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Estabilidad por repetición
        if predicted_character == current_stable_prediction:
            consecutive_count += 1
        else:
            consecutive_count = 1
            current_stable_prediction = predicted_character

        if consecutive_count >= STABILITY_THRESHOLD and predicted_character != last_prediction:
            accumulated_text += predicted_character
            with open("resultado.txt", "a") as f:
                f.write(predicted_character)
            last_prediction = predicted_character

    # Mostrar texto acumulado en la interfaz
    label_text.config(text=accumulated_text)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.imgtk = imgtk
    canvas.configure(image=imgtk)

    root.after(10, update_frame)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()