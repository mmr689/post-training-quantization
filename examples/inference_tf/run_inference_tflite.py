# """ tflite"""

import cv2
import numpy as np
import tensorflow as tf
import time

def run_inference(image_path, model_path, labels_path):
    # 1. Medir el tiempo de carga del modelo
    start_load_time = time.time()
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    load_time = time.time() - start_load_time

    # 2. Obtener detalles de las entradas y salidas
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. Cargar las etiquetas
    with open(labels_path, 'r') as f:
        labels = eval(f.read())  # Suponiendo que el archivo tiene una lista de etiquetas

    # 4. Cargar y preprocesar la imagen
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))  # Cambiar el tamaño según el modelo
    img_normalized = img_resized.astype(np.float32) / 255.0  # Normalizar entre 0 y 1
    batched_img = np.expand_dims(img_normalized, axis=0)  # Añadir una dimensión para el batch

    # 5. Medir el tiempo de inferencia
    start_inference_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], batched_img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    inference_time = time.time() - start_inference_time

    # 6. Calcular probabilidades con NumPy
    output_data = output_data[0]  # Acceder a la primera dimensión si es un batch
    exp_scores = np.exp(output_data - np.max(output_data))  # Estabilidad numérica
    probabilidades = exp_scores / np.sum(exp_scores)

    # Obtener las 5 probabilidades más altas y sus índices
    top_5_indices = np.argsort(probabilidades)[::-1][:5]

    # Mostrar las 5 etiquetas con las mayores probabilidades
    print()
    for i, idx in enumerate(top_5_indices):
        certainty_percentage = probabilidades[idx] * 100
        print(f"Top {i + 1}: {labels[idx]} ({idx}) con un {certainty_percentage:.2f}% de certeza.")

    # 7. Imprimir tiempos
    print(f"\nTiempo de carga del modelo: {load_time:.4f} segundos")
    print(f"Tiempo de inferencia: {inference_time:.4f} segundos")
    print(f"Tiempo total: {load_time + inference_time:.4f} segundos")

    return load_time, inference_time


if __name__ == "__main__":
    image_path = 'assets/dog.jpg'  # Ruta de la imagen
    model_path = 'results/resnet_imagenet-default/resnet.tflite'  # Ruta del modelo TFLite
    labels_path = 'assets/imagenet-simple-labels.txt'  # Ruta de las etiquetas

    total_load_time = 0
    total_inference_time = 0
    iterations = 10  # Definir cuántas veces ejecutar la inferencia

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
