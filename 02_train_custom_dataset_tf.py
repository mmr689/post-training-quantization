import os
import shutil
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from utils.model_quantizer import ModelQuantizer

# Parámetros
MODEL_NAME = 'mobilenet'  # Cambia a 'mobilenet' o 'resnet' si es necesario
IMG_SIZE = (224, 224)  # Tamaño de las imágenes
BATCH_SIZE = 32
EPOCHS = 2  # Número de épocas
LEARNING_RATE = 0.0001  # Tasa de aprendizaje
DATA_DIR = "data"
TRAIN_TXT_PATH = "data/train.txt"
VAL_TXT_PATH = "data/validation.txt"
LABELS = ['rice_weevil', 'red_flour_beetle']  # Ajusta las etiquetas según tu dataset
OUTPUT_DIR = f"results/{MODEL_NAME}_trap-colour-insects"

# Crear directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cargar_datos_y_etiquetas(txt_path, labels, img_size):
    """
    Carga rutas de imágenes y etiquetas desde un archivo .txt, y convierte las imágenes a arrays.
    
    Args:
    txt_path (str): Ruta del archivo .txt con las rutas de las imágenes.
    labels (list): Lista de etiquetas correspondientes.
    img_size (tuple): Tamaño deseado para las imágenes.

    Returns:
    tuple: Arrays de imágenes y etiquetas correspondientes.
    """
    rutas_imagenes = []
    etiquetas = []

    with open(txt_path, 'r') as f:
        for line in f:
            ruta_relativa = line.strip()
            ruta_completa = os.path.join(DATA_DIR, ruta_relativa)

            for i, label in enumerate(labels):
                if label in ruta_relativa:
                    etiqueta = i
                    break

            rutas_imagenes.append(ruta_completa)
            etiquetas.append(etiqueta)

    imagenes = [img_to_array(load_img(ruta, target_size=img_size)) / 255.0 for ruta in rutas_imagenes]
    return np.array(imagenes), np.array(etiquetas)


def load_pretrained_model(model_name, input_shape):
    """
    Carga un modelo preentrenado según el nombre especificado.
    
    Args:
    model_name (str): Nombre del modelo ('vgg16', 'mobilenet', 'resnet').
    input_shape (tuple): Forma de las imágenes de entrada.

    Returns:
    Model: Modelo base preentrenado sin la capa superior.
    """
    if model_name == 'vgg16':
        return tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'mobilenet':
        return tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'resnet':
        return tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Modelo no soportado. Usa 'vgg16', 'mobilenet' o 'resnet'.")


def build_model(base_model, num_classes):
    """
    Añade capas superiores a un modelo base y lo devuelve listo para entrenamiento.
    
    Args:
    base_model (Model): Modelo preentrenado sin la capa superior.
    num_classes (int): Número de clases de salida.

    Returns:
    Model: Modelo completo listo para ser entrenado.
    """
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


def plot_metrics(history, output_dir):
    """
    Genera gráficos de precisión y pérdida durante el entrenamiento.
    
    Args:
    history (History): Historial del entrenamiento.
    output_dir (str): Directorio donde se guardarán los gráficos.
    """
    plt.figure(figsize=(12, 4))

    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid()

    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metricas_entrenamiento.png'))
    plt.close()


def save_metrics(history, output_dir):
    """
    Guarda las métricas del entrenamiento en un archivo JSON.
    
    Args:
    history (History): Historial del entrenamiento.
    output_dir (str): Directorio donde se guardará el archivo JSON.
    """
    metrics_path = os.path.join(output_dir, 'metricas_entrenamiento.json')
    with open(metrics_path, 'w') as json_file:
        json.dump(history.history, json_file)


def quantize_and_save(model_path, repr_dataset_path, output_dir, model_name):
    """
    Aplica varias técnicas de cuantización y guarda los modelos cuantizados.
    
    Args:
    model_path (str): Ruta del modelo guardado.
    repr_dataset_path (str): Ruta al dataset representativo para la cuantización.
    output_dir (str): Directorio donde se guardarán los modelos cuantizados.
    model_name (str): Nombre del modelo base.
    """
    quantizer = ModelQuantizer(model_path)

    # Cuantización dinámica
    dynamic_quant_model = quantizer.quantize_dynamic_range()
    quantizer.save_quantized_model(dynamic_quant_model, os.path.join(output_dir, f'{model_name}_dynamic_quant.tflite'))

    # Cuantización float16
    float16_quant_model = quantizer.quantize_float16()
    quantizer.save_quantized_model(float16_quant_model, os.path.join(output_dir, f'{model_name}_float16_quant.tflite'))

    # Cuantización solo enteros
    integer_only_quant_model = quantizer.quantize_integer_only(repr_dataset_path, (224, 224))
    quantizer.save_quantized_model(integer_only_quant_model, os.path.join(output_dir, f'{model_name}_integer_only_quant.tflite'))

    # Cuantización completa de enteros
    full_integer_quant_model = quantizer.quantize_full_integer(repr_dataset_path, (224, 224))
    quantizer.save_quantized_model(full_integer_quant_model, os.path.join(output_dir, f'{model_name}_full_integer_quant.tflite'))


# Cargar y preparar datos
x_train, y_train = cargar_datos_y_etiquetas(TRAIN_TXT_PATH, LABELS, IMG_SIZE)
x_val, y_val = cargar_datos_y_etiquetas(VAL_TXT_PATH, LABELS, IMG_SIZE)
y_train = to_categorical(y_train, num_classes=len(LABELS))
y_val = to_categorical(y_val, num_classes=len(LABELS))

# Cargar el modelo preentrenado
base_model = load_pretrained_model(MODEL_NAME, (IMG_SIZE[0], IMG_SIZE[1], 3))

# Construir y entrenar el modelo
model = build_model(base_model, num_classes=len(LABELS))
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

# Guardar métricas y gráficos
save_metrics(history, OUTPUT_DIR)
plot_metrics(history, OUTPUT_DIR)

# Guardar el modelo en formato H5 y SavedModel
model_h5_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_reentrenado.h5")
model.save(model_h5_path)
saved_model_path = os.path.join(OUTPUT_DIR, f"saved_model")
tf.keras.models.save_model(model, saved_model_path, save_format='tf')

# Crear directorio temporal para almacenar las imágenes representativas
REPR_DATASET_PATH = "temp_repr_dataset"
os.makedirs(REPR_DATASET_PATH, exist_ok=True)

# Guardar las imágenes de validación en la carpeta temporal
with open(VAL_TXT_PATH, 'r') as f:
    for idx, line in enumerate(f):
        ruta_relativa = line.strip()
        _, extension = os.path.splitext(ruta_relativa)  # Obtener la extensión original del archivo
        img_path = os.path.join(REPR_DATASET_PATH, f"val_img_{idx}{extension}")
        save_img(img_path, x_val[idx])  # Guardar la imagen en el mismo formato original

# Aplicar cuantización y guardar los modelos cuantizados
quantize_and_save(saved_model_path, REPR_DATASET_PATH, OUTPUT_DIR, MODEL_NAME)

# Eliminar el directorio temporal una vez que se haya completado la cuantización
shutil.rmtree(REPR_DATASET_PATH)