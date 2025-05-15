import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Ruta a tu dataset
DATASET_PATH = r"E:\3PrimerCiclo2025\IA\ProyectoFinal-ia\asl_alphabet_train"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

# Aumento de datos
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Modelo base MobileNetV2
base_model = MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compilar
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Guardar modelo
checkpoint = ModelCheckpoint(
    "modelo_asl.h5", save_best_only=True, monitor="val_accuracy", mode="max")
model.fit(train_generator, validation_data=val_generator,
          epochs=EPOCHS, callbacks=[checkpoint])

print("\n✅ Entrenamiento finalizado.")
