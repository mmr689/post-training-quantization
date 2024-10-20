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


# Download the pre-trained MobileNet model with ImageNet weights
model = tf.keras.applications.MobileNet(weights='imagenet')

# Save the model in TensorFlow SavedModel format
saved_model_path = "results/TEST_mobilenet_tf_saved_model"
tf.saved_model.save(model, saved_model_path)

# Load the saved TensorFlow model from the specified directory
loaded_model = tf.saved_model.load(saved_model_path)

# Check and print the available signatures of the loaded model
if 'serving_default' in loaded_model.signatures:
    infer = loaded_model.signatures['serving_default']
    print("Details of the 'serving_default' signature:")
    print(" - Inputs:", infer.structured_input_signature)
    print(" - Outputs:", infer.structured_outputs)
else:
    print("The 'serving_default' signature is not available in this model.")

# Setup the TFLite converter for post-training quantization
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
repr_dataset_path = '/home/qcienmed/mmr689/quantization/val2017'
converter.representative_dataset = lambda: representative_dataset(folder_path= repr_dataset_path, size=(224, 224))

# Convert the model to a quantized TensorFlow Lite model
tflite_quant_model = converter.convert()

# Save the quantized TFLite model to a file
quantized_model_path = saved_model_path + '/full_integer_quant.tflite'
with open(quantized_model_path, 'wb') as f:
    f.write(tflite_quant_model)

print('Quantized model saved at:', quantized_model_path)