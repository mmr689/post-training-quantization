"""
Este código analiza para las 4 cuantizaciones una única imagen. 
"""

import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
import time
import json

def load_model(model_path):
    # 1. Medir el tiempo de carga del modelo
    start_load_time = time.time()
    interpreter = Interpreter(model_path, experimental_delegates=[load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    load_time = time.time() - start_load_time

    return interpreter, load_time

def run_inference_Int8Quant(image_path, interpreter, n_probs=5):

    # 2. Obtener detalles de las entradas y salidas
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]
    output_scale = output_details[0]['quantization'][0]
    output_zero_point = output_details[0]['quantization'][1]

    # 4. Cargar y preprocesar la imagen
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))  # Cambiar el tamaño según el modelo
    img_normalized = img_resized.astype(np.float32) / 255.0  # Normalizar la imagen

    # 4.1. Cuantización inversa: aplicar escala y punto cero
    img_quantized = img_normalized / input_scale  # Escala inversa
    img_quantized = img_quantized - input_zero_point  # Aplicar el punto cero
    img_quantized = img_quantized.astype(np.int8)

    # 4.2. Añadir la dimensión de batch
    batched_img = np.expand_dims(img_quantized, axis=0)

    # 5. Medir el tiempo de inferencia
    start_inference_time = time.time()
    interpreter.set_tensor(input_index, batched_img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)
    inference_time = time.time() - start_inference_time

     # 6. Deshacer la cuantización: aplicar escala y punto cero a la salida
    output_float = (output_data - output_zero_point) * output_scale
    output_float = np.squeeze(output_float)  # Eliminar la dimensión de lote

    # 7. Aplicar softmax para convertir los puntajes en probabilidades
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Estabilidad numérica
        return exp_x / np.sum(exp_x)
    
    probs = softmax(output_float)
    # Obtener las n_probs más altas de clasificación
    top_5_indices = np.argsort(probs)[::-1][:n_probs]
    top_5_probs = probs[top_5_indices]

    return inference_time, [top_5_indices, top_5_probs]

if __name__ == "__main__":
    main_path = 'results/yolov8n_trap-colour-insects'
    labels_path = os.path.join(main_path, 'labels.txt')  # Ruta de las etiquetas
    path_base = 'data/trap-colour-insects-dataset' # Define el path base imágenes validación

    # Lee el archivo de validación
    with open('data/trap-colour-insects-dataset/validation.txt', 'r') as f:
        rutas_relativas = [line.strip() for line in f]
    # Genera la lista final uniendo el path base con las rutas relativas
    images_paths = [os.path.join(path_base, ruta) for ruta in rutas_relativas]
    # images_paths = ['data/trap-colour-insects-dataset/red_flour_beetle/0_1.jpg']
    
    # Cargar las etiquetas
    with open(labels_path, 'r') as f:
        labels = eval(f.read())

    # ********************************* #
    #         Full INT Quant            #
    # ********************************* #
    model_name = 'best_full_integer_quant'
    model_path = os.path.join(main_path, f'{model_name}.tflite')  # Ruta del modelo TFLite
    interpreter, load_time = load_model(model_path)
    metrics = {'load time': load_time}
    for cnt, image_path in enumerate(images_paths):
        inference_time, results = run_inference_Int8Quant(image_path, interpreter)
        metrics[cnt] = {
            'image name': image_path.split('/')[-1],
            'label name': image_path.split('/')[-2],
            'inference time': inference_time,
            'inference ids': results[0].tolist(),
            'inference probs': results[1].tolist()
            }
    
    # Guardar el diccionario en un archivo JSON
    json.dump(metrics, open(os.path.join(main_path, f'{model_name}_edge_inference_metrics.json'), 'w'), indent=4)

    # ********************************* #
    #         INT Only Quant            #
    # ********************************* #
    model_name = 'best_full_integer_quant_edgetpu'
    model_path = os.path.join(main_path, f'{model_name}.tflite')  # Ruta del modelo TFLite
    interpreter, load_time = load_model(model_path)
    metrics = {'load time': load_time}
    for cnt, image_path in enumerate(images_paths):
        inference_time, results = run_inference_Int8Quant(image_path, interpreter)
        metrics[cnt] = {
            'image name': image_path.split('/')[-1],
            'label name': image_path.split('/')[-2],
            'inference time': inference_time,
            'inference ids': results[0].tolist(),
            'inference probs': results[1].tolist()
            }
    
    # Guardar el diccionario en un archivo JSON
    json.dump(metrics, open(os.path.join(main_path, f'{model_name}_edge_inference_metrics.json'), 'w'), indent=4)