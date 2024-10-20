"""
Float16 quantization
https://www.tensorflow.org/lite/performance/post_training_quantization#float16_quantization
"""

import tensorflow as tf

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
converter.target_spec.supported_types = [tf.float16]

# Convert the model to a quantized TensorFlow Lite model
tflite_quant_model = converter.convert()

# Save the quantized TFLite model to a file
quantized_model_path = saved_model_path + '/float16_quant.tflite'
with open(quantized_model_path, 'wb') as f:
    f.write(tflite_quant_model)

print('Quantized model saved at:', quantized_model_path)