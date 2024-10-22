import tensorflow as tf
from PIL import Image
import numpy as np
import os

class ModelQuantizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.converter = None
    
    def load_and_preprocess_image(self, path, size):
        """Load and preprocess an image from the given path and resize to the given size."""
        image = Image.open(path)
        image = image.resize(size)  # Resize to model input size
        if image.mode != 'RGB':
            image = image.convert('RGB')  # Convert grayscale to RGB
        image = np.array(image) / 255.0   # Model normalization
        return image.astype(np.float32)

    def representative_dataset(self, folder_path, size):
        """Generate a representative dataset from images in a specified folder."""        
        images = [self.load_and_preprocess_image(os.path.join(folder_path, filename), size)
                  for filename in os.listdir(folder_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image in images:
            yield [np.expand_dims(image, axis=0)]

    def quantize_dynamic_range(self):
        """Quantize the model using dynamic range quantization."""
        self.converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = self.converter.convert()
        return tflite_model

    def quantize_float16(self):
        """Quantize the model using Float16 quantization."""
        self.converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.converter.target_spec.supported_types = [tf.float16]
        tflite_model = self.converter.convert()
        return tflite_model

    def quantize_integer_only(self, repr_dataset_path, size):
        """Quantize the model using integer only quantization with representative data."""
        self.converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.converter.representative_dataset = lambda: self.representative_dataset(repr_dataset_path, size)
        self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        self.converter.inference_input_type = tf.int8
        self.converter.inference_output_type = tf.int8
        tflite_model = self.converter.convert()
        return tflite_model

    def quantize_full_integer(self, repr_dataset_path, size):
        """Quantize the model using full integer quantization, including float fallback."""
        self.converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.converter.representative_dataset = lambda: self.representative_dataset(repr_dataset_path, size)
        self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        self.converter.inference_input_type = tf.int8
        self.converter.inference_output_type = tf.int8
        tflite_model = self.converter.convert()
        return tflite_model

    def save_quantized_model(self, tflite_model, output_path):
        """Save the quantized TFLite model to the specified output path."""
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"\n · Quantized model saved to {output_path}\n")
