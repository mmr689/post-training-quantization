"""integer_only"""
import cv2
import numpy as np
import tensorflow as tf

image_path = 'assets/dog.jpg'
model_path='results/mobilenet/tf_saved_model/mobilenet_integer_only_quant.tflite'

# 1. Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

# 2. Obtener detalles de las entradas y salidas
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Cargar las etiquetas
with open('assets/imagenet-simple-labels.txt', 'r') as f:
    labels = eval(f.read())  # Suponiendo que tu archivo tiene una lista de etiquetas como el ejemplo que diste

# 4.1. Cargar y preprocesar la imagen
img = cv2.imread(image_path)
img_resized = cv2.resize(img, (224, 224))
img_normalized = img_resized.astype(np.float32) / 255.0 # Normalizar la imagen al rango [0, 255]

# 4.2. Cuantización inversa: aplicar escala y punto cero
img_quantized = img_normalized / 0.003921568859368563  # Aplicar la escala (1/255)
img_quantized = img_quantized - (-128)  # Aplicar el punto cero
img_quantized = img_quantized.astype(np.int8)

# 4.3. Añadir la dimensión de lote
batched_img = np.expand_dims(img_quantized, axis=0)

# 5. Realizar la predicción
interpreter.set_tensor(input_details[0]['index'], batched_img)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Obtener la salida cuantizada del modelo
output_data = interpreter.get_tensor(output_details[0]['index'])

# Deshacer la cuantización: aplicar escala y punto cero
output_float = (output_data - (-128)) * 0.00390625

# Como la forma de la salida es (1, 1000), eliminamos la dimensión de lote
output_float = np.squeeze(output_float)

# Imprimir las 5 clases con mayor puntaje (logit)
top_5_indices = np.argsort(output_float)[-5:][::-1]
print("\nTop 5 clases con mayor puntaje:")
for i in top_5_indices:
    print(f"Clase {labels[i]} ({i}): {output_float[i]}")

# Aplicar softmax para convertir puntajes en probabilidades
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Estabilidad numérica
    return exp_x / np.sum(exp_x)

# Aplicar softmax a la salida float
probabilidades = softmax(output_float)
# Imprimir las 5 clases con mayor probabilidad
top_5_indices = np.argsort(probabilidades)[-5:][::-1]
print("\nTop 5 clases con mayor probabilidad:")
for i in top_5_indices:
    print(f"Clase {i}: {probabilidades[i]*100}")