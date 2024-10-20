"""
Código Coral
"""

from tflite_runtime.interpreter import Interpreter, load_delegate
import os
import cv2
import numpy as np
import time


# Ruta al modelo TFLite
model_path = 'bioview/yolov8n_full_integer_quant_edgetpu.tflite'
interpreter = Interpreter(model_path, experimental_delegates=[load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()


# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']

image_path = 'assets/dog.jpg'
frame = cv2.imread(image_path)
resized_img = cv2.resize(frame, (width, height))
# Normalizar los valores de píxeles a INT8
norm_img = resized_img.astype(np.int8)
# Agregar una dimensión para representar el lote (batch)
batched_img = np.expand_dims(norm_img, axis=0)


# Perform the actual detection by running the model with the image as input
start_time = time.time()
interpreter.set_tensor(input_details[0]['index'], batched_img)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])