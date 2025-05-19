# 📸 Detector de Lenguaje de Señas (Alfabeto ASL)

 Capturar, entrenar y reconocer letras del lenguaje de señas americano (ASL) en tiempo real usando Python, OpenCV y MediaPipe.

## 📦 Requisitos

1. **Python 3.10.x**
2. Dependencias:
pip install opencv-python mediapipe scikit-learn numpy pillow seaborn matplotlib 
3. Uso: 
    1. Capturar el Dataset
        Ejecuta TomandoDatos.py para capturar imágenes de las señas del alfabeto (A-Z):
    2. Procesar el Data set
        Ejecuta CreadorDeDataset.py para extraer características de las imágenes y generar data.pickle
    3. Entranar el modelo
        Ejecuta EntrenarModelo.py para entrenar un clasificador Random Forest
    4. Usar interfaz en tiempo real
        Ejecuta Camara.py para detectar señas en tiempo real
4. Dataset propio utilizado para el entrenamiento del modelo: 
https://drive.google.com/file/d/13VATcKw8D2mQ8mawv3jgg5mm72k2OkNX/view?usp=sharing
4. Crear y activar un entorno virtual:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

---Si no funciona requirements: 
-Descargar Repositorio.
-Abrir Visual Studio -- abrir carpeta
-Nueva terminal--->Crear Entorno virtual--> python -m venv venv
-Terminal para activar entorno: .\venv\Scripts\Activate.ps1 
-Terminal librerias: pip install opencv-python==4.7.0.68 mediapipe scikit-learn==1.2.0






