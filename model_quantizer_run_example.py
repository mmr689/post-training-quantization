import os
import tensorflow as tf
from utils.model_quantizer import ModelQuantizer

# Download the pre-trained MobileNet model with ImageNet weights
model = tf.keras.applications.VGG16(weights='imagenet')

# Save the model in TensorFlow SavedModel format
saved_model_path = "results/vgg16_tf_saved_model"
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
quantizer.save_quantized_model(dynamic_quant_model, os.path.join(saved_model_path, 'mobilenet_dynamic_quant.tflite'))

# Apply float16 quantization
float16_quant_model = quantizer.quantize_float16()
quantizer.save_quantized_model(float16_quant_model, os.path.join(saved_model_path, 'mobilenet_float16_quant.tflite'))

# Apply integer only quantization
integer_only_quant_model = quantizer.quantize_integer_only(repr_dataset_path, (224, 224))
quantizer.save_quantized_model(integer_only_quant_model, os.path.join(saved_model_path, 'mobilenet_integer_only_quant.tflite'))

# Apply full integer quantization
full_integer_quant_model = quantizer.quantize_full_integer(repr_dataset_path, (224, 224))
quantizer.save_quantized_model(full_integer_quant_model, os.path.join(saved_model_path, 'mobilenet_full_integer_quant.tflite'))