"""
Código que ejecuta el modelo si cuantizar .h5
"""

import cv2
import numpy as np
import tensorflow as tf
import time

def run_inference(image_path, model_path, labels_path):
    # 1. Medir el tiempo de carga del modelo
    start_load_time = time.time()
    model = tf.keras.models.load_model(model_path)
    load_time = time.time() - start_load_time

    # 2. Cargar las etiquetas
    with open(labels_path, 'r') as f:
        labels = eval(f.read())  # Suponiendo que tu archivo tiene una lista de etiquetas

    # 3. Cargar y preprocesar la imagen
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))  # Redimensionar la imagen según el modelo
    img_normalized = img_resized / 255.0  # Normalizar entre 0 y 1
    img_expanded = np.expand_dims(img_normalized, axis=0)  # Añadir una dimensión para el batch

    # 4. Medir el tiempo de inferencia
    start_inference_time = time.time()
    predictions = model.predict(img_expanded)[0]
    inference_time = time.time() - start_inference_time

    # Ordenar las predicciones y obtener las 5 mejores
    top_5_indices = np.argsort(predictions)[-5:][::-1]

    # Mostrar las 5 mejores predicciones con sus porcentajes de certeza
    print()
    for i, idx in enumerate(top_5_indices):
        certainty_percentage = predictions[idx] * 100
        print(f"Top {i + 1}: {labels[idx]} ({idx}) con un {certainty_percentage:.2f}% de certeza.")

    # 5. Imprimir tiempos
    print(f"\nTiempo de carga del modelo: {load_time:.4f} segundos")
    print(f"Tiempo de inferencia: {inference_time:.4f} segundos")
    print(f"Tiempo total: {load_time + inference_time:.4f} segundos")

    return load_time, inference_time


if __name__ == "__main__":
    image_path = 'data/rice_weevil/21_37.jpg'             # 'assets/dog.jpg' | 'data/rice_weevil/21_37.jpg'
    model_path = 'results/vgg16_trap-colour-insects/vgg16_reentrenado.h5'
    labels_path = 'assets/imagenet-simple-labels.txt'

    total_load_time = 0
    total_inference_time = 0
    iterations = 10

    # Ejecutar el bucle {iterations} veces
    for i in range(iterations):
        print(f"\nEjecución {i + 1}:\n")
        load_time, inference_time = run_inference(image_path, model_path, labels_path)
        total_load_time += load_time
        total_inference_time += inference_time

    # Calcular el promedio
    avg_load_time = total_load_time / iterations
    avg_inference_time = total_inference_time / iterations

    # Imprimir los tiempos promedios
    print(f"\nPromedio de tiempo de carga del modelo: {avg_load_time:.4f} segundos")
    print(f"Promedio de tiempo de inferencia: {avg_inference_time:.4f} segundos")
    print(f"Promedio de tiempo total: {avg_load_time + avg_inference_time:.4f} segundos")