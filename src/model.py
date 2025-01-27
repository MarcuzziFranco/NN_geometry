import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore

def create_model(input_shape=(64, 64, 1), num_classes=3):
    model = Sequential([
        # Capa 1: Convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        # Capa 2: Convolucional
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Capa 3: Aplanar y densa
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # Salida con 3 clases
    ])
    return model

if __name__ == "__main__":
    model = create_model()
    model.summary()  # Muestra la arquitectura del modelo
