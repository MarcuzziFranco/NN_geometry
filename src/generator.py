import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
import random
import argparse

# Configuración de parámetros
IMG_SIZE = 64
CLASSES = ["circles", "squares", "triangles"]

# Asegurarse de que las carpetas existan
def create_directories(output_dir):
    for class_name in CLASSES:
        class_path = os.path.join(output_dir, class_name)
        os.makedirs(class_path, exist_ok=True)

# Función para generar un círculo con aleatoriedad
def generate_circle():
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    center = (random.randint(10, IMG_SIZE - 10), random.randint(10, IMG_SIZE - 10))
    radius = random.randint(5, IMG_SIZE // 4)
    y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    img[mask] = 255
    return img

# Función para generar un cuadrado con aleatoriedad
def generate_square():
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    size = random.randint(10, IMG_SIZE // 2)
    start_x = random.randint(0, IMG_SIZE - size)
    start_y = random.randint(0, IMG_SIZE - size)
    img[start_y:start_y + size, start_x:start_x + size] = 255
    return img

# Función para generar un triángulo con aleatoriedad
def generate_triangle():
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    x1, y1 = random.randint(5, IMG_SIZE - 5), random.randint(5, IMG_SIZE // 2)
    x2, y2 = random.randint(5, IMG_SIZE - 5), random.randint(IMG_SIZE // 2, IMG_SIZE - 5)
    x3, y3 = random.randint(5, IMG_SIZE - 5), random.randint(IMG_SIZE // 2, IMG_SIZE - 5)
    # Coordenadas del triángulo
    r = np.array([y1, y2, y3])
    c = np.array([x1, x2, x3])
    rr, cc = polygon(r, c, img.shape)
    img[rr, cc] = 255
    return img

# Función para guardar imágenes
def save_images(output_dir, class_name, generate_func, num_images):
    for i in range(num_images):
        img = generate_func()
        file_path = os.path.join(output_dir, class_name, f"{class_name}_{i}.png")
        plt.imsave(file_path, img, cmap='gray')

# Generar las imágenes
def generate_data(output_dir, num_images):
    create_directories(output_dir)
    save_images(output_dir, "circles", generate_circle, num_images)
    save_images(output_dir, "squares", generate_square, num_images)
    save_images(output_dir, "triangles", generate_triangle, num_images)

# Punto de entrada principal
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de datos de formas geométricas.")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data/train/", 
        help="Directorio de salida donde se guardarán las imágenes."
    )
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=1000, 
        help="Cantidad de imágenes a generar por categoría."
    )
    args = parser.parse_args()

    print(f"Generando datos en: {args.output_dir}")
    print(f"Número de imágenes por categoría: {args.num_images}")
    generate_data(args.output_dir, args.num_images)
    print("¡Datos generados exitosamente!")
