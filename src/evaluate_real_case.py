import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from skimage.measure import label, regionprops

# Configuración
MODEL_PATH = "./models/shape_classifier.h5"
CLASSES = ["circles", "squares", "triangles"]

# Función para preprocesar la imagen real
def preprocess_image(image_path):
    # Cargar la imagen en escala de grises
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    # Aplicar filtro gaussiano para suavizar la imagen
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Binarización (umbral adaptativo)
    _, binary = cv2.threshold(img_blurred, 128, 255, cv2.THRESH_BINARY_INV)

    # Eliminar ruido y cuadrícula con operaciones morfológicas
    kernel = np.ones((3, 3), np.uint8)
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Etiquetar regiones conectadas (figuras)
    labeled_img = label(binary_cleaned)
    regions = regionprops(labeled_img)

    # Filtrar regiones por tamaño y ordenar por área
    regions = [region for region in regions if region.area > 500]
    regions = sorted(regions, key=lambda r: r.area, reverse=True)[:3]  # Mantener solo las 3 más grandes

    # Extraer figuras y normalizarlas
    figures = []
    filtered_regions = []
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        figure = binary_cleaned[min_row:max_row, min_col:max_col]

        # Redimensionar al tamaño esperado por el modelo (64x64)
        #resized_figure = cv2.resize(figure, (64, 64), interpolation=cv2.INTER_AREA)
        resized_figure = resize_with_padding(figure)

        # Normalizar a valores entre 0 y 1
        resized_figure = resized_figure.astype("float32") / 255.0
        resized_figure = np.expand_dims(resized_figure, axis=-1)  # Añadir canal
        figures.append(resized_figure)
        filtered_regions.append(region)

    return np.array(figures), img, binary_cleaned, filtered_regions



def preprocess_image_simple(image_path):
    # Cargar la imagen en escala de grises
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    # Invertir colores (figuras negras sobre fondo blanco)
    #img = cv2.bitwise_not(img)

    # Binarización simple (umbral fijo)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Etiquetar regiones conectadas (figuras)
    labeled_img = label(binary)
    regions = regionprops(labeled_img)

    # Filtrar regiones por tamaño y mantener solo las más grandes
    regions = [region for region in regions if region.area > 500]
    regions = sorted(regions, key=lambda r: r.area, reverse=True)

    # Extraer figuras y normalizarlas
    figures = []
    filtered_regions = []
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        figure = binary[min_row:max_row, min_col:max_col]

        # Redimensionar al tamaño esperado por el modelo (64x64) 
        #resized_figure = resize_with_padding(figure)

        # Normalizar a valores entre 0 y 1
        #resized_figure = resized_figure.astype("float32") / 255.0
        #resized_figure = np.expand_dims(resized_figure, axis=-1)  # Añadir canal

        # Centrar la figura en un fondo negro
        centered_figure = center_figure_in_canvas(figure)

        # Normalizar y agregar al conjunto de figuras
        centered_figure = centered_figure.astype("float32") / 255.0
        centered_figure = np.expand_dims(centered_figure, axis=-1)  # Añadir canal
        figures.append(centered_figure)
        filtered_regions.append(region)

        
    visualize_detected_figures([fig[:, :, 0] for fig in figures])
    return np.array(figures), img, binary, filtered_regions

def resize_with_padding(figure, target_size=(64, 64)):
    # Crear un lienzo cuadrado (negro) del tamaño objetivo
    canvas = np.zeros(target_size, dtype=np.uint8)

    # Obtener dimensiones originales
    h, w = figure.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Redimensionar la figura manteniendo proporciones
    resized = cv2.resize(figure, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Colocar la figura centrada en el lienzo
    top = (target_size[0] - new_h) // 2
    left = (target_size[1] - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def center_figure_in_canvas(figure, target_size=(64, 64), scale_factor=0.7):
    """
    Centra una figura en un fondo negro del tamaño especificado, la escala y aplica binarización.

    Parameters:
    - figure: np.array, la figura recortada.
    - target_size: tuple, tamaño del lienzo de salida (por defecto 64x64).
    - scale_factor: float, factor para reducir el tamaño de la figura (0.0 - 1.0).
    """
    # Crear un lienzo negro (canvas)
    canvas = np.zeros(target_size, dtype=np.uint8)

    # Obtener dimensiones originales de la figura
    h, w = figure.shape
    scale = min(target_size[0] / h, target_size[1] / w) * scale_factor  # Escala ajustada
    new_h, new_w = int(h * scale), int(w * scale)

    # Redimensionar la figura manteniendo proporciones
    resized_figure = cv2.resize(figure, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calcular las coordenadas para centrar la figura en el lienzo
    top = (target_size[0] - new_h) // 2
    left = (target_size[1] - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized_figure

    # Binarizar la imagen
    _, binary_canvas = cv2.threshold(canvas, 128, 255, cv2.THRESH_BINARY)

    return binary_canvas

def visualize_detected_figures(figures):
    """
    Muestra las figuras detectadas antes de enviarlas al modelo.
    """
    plt.figure(figsize=(10, 5))
    for i, figure in enumerate(figures):
        plt.subplot(1, len(figures), i + 1)
        plt.imshow(figure, cmap="gray")
        plt.title(f"Figura {i + 1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# Función para realizar predicciones
def predict_figures(model, figures):
    predictions = model.predict(figures)
    predicted_classes = [CLASSES[np.argmax(pred)] for pred in predictions]
    confidences = [np.max(pred) for pred in predictions]
    return predicted_classes, confidences

# Visualizar los resultados
def visualize_predictions(image, filtered_regions, predictions, confidences):
    # Crear una copia de la imagen original para dibujar
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Dibujar cada figura con su predicción
    for region, pred, conf in zip(filtered_regions, predictions, confidences):
        min_row, min_col, max_row, max_col = region.bbox
        cv2.rectangle(output_image, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)
        cv2.putText(
            output_image, f"{pred} ({conf:.2f})",
            (min_col, min_row - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    # Mostrar la imagen con predicciones
    plt.figure(figsize=(10, 10))
    plt.imshow(output_image[..., ::-1])  # Convertir BGR a RGB para matplotlib
    plt.title("Predicciones Refinadas")
    plt.axis("off")
    plt.show()




try:
    if __name__ == "__main__":
        # Ruta a la imagen real
        image_path = "./data/real/Imagen_paint_2.png"  # Cambia esto por el nombre de tu archivo

        # Cargar el modelo entrenado
        model = tf.keras.models.load_model(MODEL_PATH) # type: ignore

        # Preprocesar la imagen
        figures, img,original_image, labeled_image = preprocess_image_simple(image_path)

        # Realizar predicciones
        predictions, confidences = predict_figures(model, figures)

        # Mostrar resultados
        visualize_predictions(img, labeled_image, predictions, confidences)
        print("Predicciones realizadas con éxito.")
except KeyboardInterrupt:
    print("\nEjecución interrumpida por el usuario.")
    exit()