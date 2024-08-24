import os
import tensorflow as tf
from utils.model_quantizer import ModelQuantizer

# Load the saved TensorFlow model from the specified directory
saved_model_path = 'results/YOLOv8n/yolov8n-cls_saved_model'
loaded_model = tf.saved_model.load(saved_model_path)

print(type(loaded_model))
tf.saved_model.save(loaded_model, saved_model_path)

loaded_model = tf.saved_model.load(saved_model_path)
print(loaded_model.signatures)  # Esto mostrará las firmas disponibles