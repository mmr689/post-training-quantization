"""
Full integer quantization: Integer with float fallback
https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization
https://www.tensorflow.org/lite/performance/post_training_quantization#integer_with_float_fallback_using_default_float_inputoutput
"""

import tensorflow as tf
import numpy as np
import os
from PIL import Image

def load_and_preprocess_image(path, size):
    image = Image.open(path)
    image = image.resize(size)  # Adaptar las imágenes al tamaño del modelo
    if image.mode != 'RGB':     # Convertir imagen en escala de grises a RGB
        image = image.convert('RGB')
    image = np.array(image) / 255.0   # Normalización del modelo
    return image.astype(np.float32)

def representative_dataset(folder_path, size):
    images = [load_and_preprocess_image(os.path.join(folder_path, filename), size) for filename in os.listdir(folder_path)]
    for image in images:
        yield [np.expand_dims(image, axis=0)]

# Configurar el convertidor
saved_model_dir = '/home/qcienmed/mmr689/quantization/PIUsatge/'
repr_dataset_path = '/home/qcienmed/mmr689/quantization/val2017'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = lambda: representative_dataset(folder_path= repr_dataset_path, size=(224, 224))

# Convertir el modelo
tflite_full_quant_model = converter.convert()

# Guardar el modelo TFLite
with open(saved_model_dir+'modelo_full_cuantificado.tflite', 'wb') as f:
    f.write(tflite_full_quant_model)