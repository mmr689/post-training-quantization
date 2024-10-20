"""full_integer"""
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

image_path = 'assets/dog.jpg'
model_path='mobilenet_integer_only_quant.tflite'
labels_path = 'assets/imagenet-simple-labels.txt'

# 1. Cargar el modelo TFLite
start_load_time = time.time()
interpreter = Interpreter(model_path, experimental_delegates=[load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()
load_time = time.time() - start_load_time

print(load_time)