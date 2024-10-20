"""
This script aims to download a pre-trained model (VGG16, MobileNet, or ResNet50) with ImageNet weights, save it in TensorFlow SavedModel format, and then apply various quantization techniques to optimize the model for deployment on resource-constrained devices. The quantization techniques applied include:

1. Dynamic Range Quantization.
2. Float16 Quantization.
3. Integer-Only Quantization using a representative dataset.
4. Full Integer Quantization with a representative dataset.

Each quantized model is saved in `.tflite` format for use on TensorFlow Lite devices. You can select the model you want to use by changing the `model_name` variable in the code.
"""

import os
import tensorflow as tf
from utils.model_quantizer import ModelQuantizer

# Function to load a pre-trained model based on user choice
def load_model(model_name):
    if model_name == 'vgg16':
        return tf.keras.applications.VGG16(weights='imagenet')
    elif model_name == 'mobilenet':
        return tf.keras.applications.MobileNetV2(weights='imagenet')
    elif model_name == 'resnet':
        return tf.keras.applications.ResNet50(weights='imagenet')
    else:
        raise ValueError("Unsupported model name. Please choose 'vgg16', 'mobilenet', or 'resnet'.")

# Choose the model name
model_name = 'vgg16'  # Change this to 'mobilenet' or 'resnet' to use other models

# Load the chosen model
model = load_model(model_name)

# Define and create the model directory if it doesn't exist
model_dir = os.path.join("results", model_name)
os.makedirs(model_dir, exist_ok=True)

# Save the model in HDF5 format
h5_model_path = os.path.join(model_dir, "{}.h5".format(model_name))
model.save(h5_model_path)

# Convert the model to TFLite format without quantization
tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
tflite_model_path = os.path.join(model_dir, "{}.tflite".format(model_name))
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# Save the model in TensorFlow SavedModel format
saved_model_path = os.path.join(model_dir, "tf_saved_model")
tf.saved_model.save(model, saved_model_path)

# Initialize the quantizer with the saved model path
quantizer = ModelQuantizer(saved_model_path)

# Path to the representative dataset
repr_dataset_path = '/home/qcienmed/mmr689/quantization/val2017'

# Function to load representative data
def representative_data_gen():
    return quantizer.representative_dataset(repr_dataset_path, size=(224, 224))

# Apply dynamic range quantization
dynamic_quant_model = quantizer.quantize_dynamic_range()
quantizer.save_quantized_model(dynamic_quant_model, os.path.join(saved_model_path, '{}_dynamic_quant.tflite'.format(model_name)))

# Apply float16 quantization
float16_quant_model = quantizer.quantize_float16()
quantizer.save_quantized_model(float16_quant_model, os.path.join(saved_model_path, '{}_float16_quant.tflite'.format(model_name)))

# Apply integer only quantization
integer_only_quant_model = quantizer.quantize_integer_only(repr_dataset_path, (224, 224))
quantizer.save_quantized_model(integer_only_quant_model, os.path.join(saved_model_path, '{}_integer_only_quant.tflite'.format(model_name)))

# Apply full integer quantization
full_integer_quant_model = quantizer.quantize_full_integer(repr_dataset_path, (224, 224))
quantizer.save_quantized_model(full_integer_quant_model, os.path.join(saved_model_path, '{}_full_integer_quant.tflite'.format(model_name)))