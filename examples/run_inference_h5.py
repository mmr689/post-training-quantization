"""
Código que ejecuta el modelo si cuantizar .h5
"""
import cv2
import numpy as np
import tensorflow as tf

# 1. Cargar el modelo
model = tf.keras.models.load_model('results/mobilenet/mobilenet.h5')

# 2. Cargar las etiquetas
with open('assets/imagenet-simple-labels.txt', 'r') as f:
    labels = eval(f.read())  # Suponiendo que tu archivo tiene una lista de etiquetas como el ejemplo que diste

# 3. Cargar y preprocesar la imagen
def load_and_preprocess_image(image_path):
    # Cargar la imagen
    img = cv2.imread(image_path)
    # Redimensionar la imagen
    img_resized = cv2.resize(img, (224, 224))  # Cambia el tamaño según el tamaño de entrada de tu modelo
    # Normalizar la imagen
    img_normalized = img_resized / 255.0  # Normaliza los valores entre 0 y 1
    # Expande las dimensiones para que sea compatible con la entrada del modelo
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded

# 4. Realizar la predicción
def predict(image_path):
    processed_image = load_and_preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_label_index = np.argmax(predictions[0])
    
    # Obtener el porcentaje de certeza
    certainty_percentage = predictions[0][predicted_label_index] * 100
    return labels[predicted_label_index], certainty_percentage

# 5. Mostrar el resultado
image_path = 'assets/dog.jpg'  # Cambia esta ruta por la ruta de tu imagen
predicted_label, certainty_percentage = predict(image_path)
print(f"La etiqueta predicha para la imagen es: {predicted_label} con un {certainty_percentage:.2f}% de certeza.")