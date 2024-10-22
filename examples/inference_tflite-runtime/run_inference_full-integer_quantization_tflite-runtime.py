"""full_integer"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

def run_inference(image_path, model_path, labels_path):
    # 1. Medir el tiempo de carga del modelo
    start_load_time = time.time()
    interpreter = tflite.Interpreter(model_path=model_path)
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
    img_normalized = img_resized.astype(np.float32) / 255.0  # Normalizar la imagen

    # 4.1. Cuantización inversa: aplicar escala y punto cero
    img_quantized = img_normalized / 0.003921568859368563  # Escala inversa (1/255)
    img_quantized = img_quantized - (-128)  # Aplicar el punto cero
    img_quantized = img_quantized.astype(np.int8)

    # 4.2. Añadir la dimensión de batch
    batched_img = np.expand_dims(img_quantized, axis=0)

    # 5. Medir el tiempo de inferencia
    start_inference_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], batched_img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    inference_time = time.time() - start_inference_time

    # 6. Deshacer la cuantización: aplicar escala y punto cero a la salida
    output_float = (output_data - (-128)) * 0.00390625
    output_float = np.squeeze(output_float)  # Eliminar la dimensión de lote

    # 7. Aplicar softmax para convertir los puntajes en probabilidades
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Estabilidad numérica
        return exp_x / np.sum(exp_x)

    probabilidades = softmax(output_float)

    # Obtener las 5 probabilidades más altas y sus índices
    top_5_indices = np.argsort(probabilidades)[-5:][::-1]
    print("\nTop 5 clases con mayor probabilidad:")
    for i in top_5_indices:
        print(f" · Clase {labels[i]} ({i}): {probabilidades[i] * 100:.2f}%")

    # 8. Imprimir tiempos
    print(f"\nTiempo de carga del modelo: {load_time:.4f} segundos")
    print(f"Tiempo de inferencia: {inference_time:.4f} segundos")
    print(f"Tiempo total: {load_time + inference_time:.4f} segundos")

    return load_time, inference_time


if __name__ == "__main__":
    image_path = 'assets/dog.jpg'  # Ruta de la imagen
    model_path = 'results/mobilenet_imagenet-default/mobilenet_full_integer_quant.tflite'  # Ruta del modelo TFLite
    labels_path = 'assets/imagenet-simple-labels.txt'  # Ruta de las etiquetas

    total_load_time = 0
    total_inference_time = 0
    iterations = 5  # Definir cuántas veces ejecutar la inferencia

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