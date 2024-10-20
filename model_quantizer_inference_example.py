import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

# Cargar el modelo TFLite
def load_tflite_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Preprocesar la imagen para la inferencia
def preprocess_image(image_path, input_size):
    image = Image.open(image_path).resize(input_size)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

# Realizar la inferencia
def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Configurar la entrada
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Ejecutar la inferencia
    interpreter.invoke()
    
    # Obtener los resultados
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Función principal de inferencia
def infer_image(model_path, image_path, input_size=(224, 224)):
    # Cargar el modelo
    interpreter = load_tflite_model(model_path)
    
    # Preprocesar la imagen
    input_data = preprocess_image(image_path, input_size)
    
    # Realizar la inferencia
    output_data = run_inference(interpreter, input_data)
    
    return output_data

# Ejemplo de uso
model_path = "results/mobilenet/tf_saved_model/mobilenet_dynamic_quant.tflite"
image_path = "assets/dog.jpg"
probabilities = infer_image(model_path, image_path)

import ast

# Abre el archivo en modo lectura
with open('assets/imagenet-simple-labels.txt', 'r') as file:
    data = file.read()

# Convierte el string en una lista de Python
labels = ast.literal_eval(data)

# Obtener los índices de las N probabilidades más altas
top_n = 5
top_n_indices = np.argsort(probabilities)[-top_n:][::-1]
top_n_indices = top_n_indices.flatten()
print(top_n_indices)
# print(labels[top_n_indices[0]])

# Mostrar las etiquetas correspondientes a los índices
print("Top {} predicciones:".format(top_n))
for idx in top_n_indices:
    label = labels[int(idx)]  # Asegúrate de que 'idx' sea un entero
    prob = probabilities[int(idx)]
    print(f"Clase: {label}, Probabilidad: {prob:.4f}")