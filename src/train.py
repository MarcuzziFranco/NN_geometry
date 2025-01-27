import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from model import create_model

# Configuración de parámetros
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = "./data/train/"
TEST_DIR = "./data/test/"

# Generador de datos
def load_data(train_dir, test_dir):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(64, 64),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    return train_data, test_data

if __name__ == "__main__":
    # Cargar los datos
    train_data, test_data = load_data(TRAIN_DIR, TEST_DIR)

    # Crear el modelo
    model = create_model()

    # Compilar el modelo
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entrenar el modelo
    model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=test_data
    )

    # Guardar el modelo entrenado
    model.save("./models/shape_classifier.h5")
    print("¡Modelo entrenado y guardado exitosamente!")
