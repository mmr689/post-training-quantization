"""full_integer"""
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

image_path = 'assets/dog.jpg'
model_path='results/mobilenet/tf_saved_model/mobilenet_float16_quant.tflite'
labels_path = 'assets/imagenet-simple-labels.txt'

# 1. Cargar el modelo TFLite
start_load_time = time.time()
interpreter = Interpreter(model_path, experimental_delegates=[load_delegate('libedgetpu.so.1', {'device': 'usb'})])
interpreter.allocate_tensors()
load_time = time.time() - start_load_time

# 2. Obtener detalles de las entradas y salidas
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Cargar las etiquetas
with open(labels_path, 'r') as f:
    labels = eval(f.read())  # Suponiendo que tu archivo tiene una lista de etiquetas como el ejemplo que diste

# 4.1. Cargar y preprocesar la imagen
img = cv2.imread(image_path)
img_resized = cv2.resize(img, (224, 224))
img_normalized = img_resized.astype(np.float32) / 255
batched_img = np.expand_dims(img_normalized, axis=0)

# 5. Realizar la predicción
start_inference_time = time.time()
interpreter.set_tensor(input_details[0]['index'], batched_img)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
inference_time = time.time() - start_inference_time

# Calcular probabilidades con NumPy
output_data = output_data[0]  # Acceder a la primera dimensión si es un batch
exp_scores = np.exp(output_data - np.max(output_data))  # Resta la máxima para estabilidad numérica
probabilidades = exp_scores / np.sum(exp_scores)

# Obtener las 5 probabilidades más altas y sus índices
top_5_indices = np.argsort(probabilidades)[::-1][:5]
top_5_probabilidades = probabilidades[top_5_indices]

# Mostrar las 5 etiquetas con las mayores probabilidades
for i, idx in enumerate(top_5_indices):
    print(f" · Clase {labels[idx]} ({idx}): {top_5_probabilidades[i] * 100:.2f}%")

print(f"\nTiempo de carga del modelo: {load_time:.4f} segundos")
print(f"Tiempo de inferencia: {inference_time:.4f} segundos")
print(f"Tiempo total: {load_time+inference_time:.4f} segundos")