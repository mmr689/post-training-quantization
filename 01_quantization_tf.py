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

# Function to create directory if not exists
def create_directory(path):
    print(path)
    os.makedirs(path, exist_ok=True)
    print(f"Directory '{path}' created successfully.")

# Function to save model in different formats
def save_model_formats(model, model_name, model_dir):
    # Save in HDF5 format
    h5_model_path = os.path.join(model_dir, "{}.h5".format(model_name))
    model.save(h5_model_path)
    print(f"\n · Model saved in HDF5 format: {h5_model_path}")
    
    # Save in TensorFlow SavedModel format
    saved_model_path = os.path.join(model_dir, "tf_saved_model")
    tf.saved_model.save(model, saved_model_path)
    print(f"\n · Model saved in TensorFlow SavedModel format: {saved_model_path}\n")
    
    return saved_model_path

# Function to convert model to TFLite without quantization
def convert_to_tflite(model, model_dir, model_name):
    tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    tflite_model_path = os.path.join(model_dir, "{}.tflite".format(model_name))
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"\n · Model saved in TFLite format: {tflite_model_path}\n")

# Function to apply different quantization methods and save them
def apply_quantization(quantizer, model_name, saved_model_path, repr_dataset_path):
    # Dynamic Range Quantization
    dynamic_quant_model = quantizer.quantize_dynamic_range()
    quantizer.save_quantized_model(dynamic_quant_model, os.path.join(saved_model_path, f'{model_name}_dynamic_quant.tflite'))

    # Float16 Quantization
    float16_quant_model = quantizer.quantize_float16()
    quantizer.save_quantized_model(float16_quant_model, os.path.join(saved_model_path, f'{model_name}_float16_quant.tflite'))

    # Integer-Only Quantization
    integer_only_quant_model = quantizer.quantize_integer_only(repr_dataset_path, (224, 224))
    quantizer.save_quantized_model(integer_only_quant_model, os.path.join(saved_model_path, f'{model_name}_integer_only_quant.tflite'))

    # Full Integer Quantization
    full_integer_quant_model = quantizer.quantize_full_integer(repr_dataset_path, (224, 224))
    quantizer.save_quantized_model(full_integer_quant_model, os.path.join(saved_model_path, f'{model_name}_full_integer_quant.tflite'))

# Main function to run the process
def main(model_name, results_dir, repr_dataset_path):
    # Load the model
    model = load_model(model_name)
    
    # Define and create the model directory if it doesn't exist
    create_directory(results_dir)

    # Save the model in different formats
    saved_model_path = save_model_formats(model, model_name, results_dir)

    # Convert the model to TFLite format without quantization
    convert_to_tflite(model, results_dir, model_name)

    # Initialize the quantizer with the saved model path
    quantizer = ModelQuantizer(saved_model_path)

    # Apply different quantization methods and save them
    apply_quantization(quantizer, model_name, results_dir, repr_dataset_path)

# User inputs
if __name__ == "__main__":
    model_name = 'mobilenet'  # Change this to 'mobilenet' or 'resnet' as needed
    results_dir = f"results/{model_name}_imagenet-default"  # Define the directory to save the results
    repr_dataset_path = 'data/val2017'  # Path to representative dataset

    # Run the process
    main(model_name, results_dir, repr_dataset_path)