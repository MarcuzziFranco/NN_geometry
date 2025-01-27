import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Configuración
TEST_DIR = "./data/test/"
BATCH_SIZE = 32
MODEL_PATH = "./models/shape_classifier.h5"
CLASSES = ["circles", "squares", "triangles"]

# Cargar los datos de prueba
def load_test_data():
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_data = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(64, 64),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    return test_data

# Evaluar el modelo
def evaluate_model():
    # Cargar el modelo entrenado
    model = tf.keras.models.load_model(MODEL_PATH) # type: ignore
    test_data = load_test_data()

    # Evaluación en los datos de prueba
    loss, accuracy = model.evaluate(test_data)
    print(f"Loss en datos de prueba: {loss:.4f}")
    print(f"Accuracy en datos de prueba: {accuracy:.4f}")

    # Predicciones
    predictions = model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_data.classes

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriz de confusión:")
    print(cm)

    # Visualización de la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicciones")
    plt.ylabel("Verdaderos")
    plt.title("Matriz de Confusión")
    plt.show()

    # Reporte de clasificación
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print("\nReporte de clasificación:")
    print(report)

if __name__ == "__main__":
    evaluate_model()
