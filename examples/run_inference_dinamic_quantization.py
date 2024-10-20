""" dinamic quantization - Funciona igual que tflite ¿son lo mismo?"""

import cv2
import numpy as np
import tensorflow as tf

image_path = 'assets/dog.jpg'
model_path='results/mobilenet/tf_saved_model/mobilenet_dynamic_quant.tflite'

# 1. Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

# 2. Obtener detalles de las entradas y salidas
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Cargar las etiquetas
with open('assets/imagenet-simple-labels.txt', 'r') as f:
    labels = eval(f.read())

# 4. Cargar y preprocesar la imagen
img = cv2.imread(image_path)
img_resized = cv2.resize(img, (224, 224))
img_normalized = img_resized.astype(np.float32) / 255
batched_img = np.expand_dims(img_normalized, axis=0)

# 5. Realizar la predicción
interpreter.set_tensor(input_details[0]['index'], batched_img)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

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