import os
import shutil
from ultralytics import YOLO

def copy_images_to_temp_folder(train_txt_path, images_base_path, output_folder):
    # Crear la carpeta de salida si no existe, no hace nada si ya existe
    os.makedirs(output_folder, exist_ok=True)

    # Leer el archivo train.txt
    with open(train_txt_path, "r") as f:
        for line in f:
            # Eliminar espacios en blanco y saltos de línea
            image_path = line.strip()
            
            # Obtener la etiqueta (nombre de la carpeta) de la imagen
            label = image_path.split("/")[0]
            
            # Ruta de la carpeta de la etiqueta dentro de la carpeta de salida
            label_folder = os.path.join(output_folder, label)
            
            # Crear la carpeta de la etiqueta si no existe
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)
            
            # Ruta completa de la imagen original
            source_path = os.path.join(images_base_path, image_path)
            
            # Ruta destino donde se copiará la imagen
            destination_path = os.path.join(label_folder, os.path.basename(image_path))
            
            # Copiar la imagen a la carpeta correspondiente
            try:
                shutil.copy(source_path, destination_path)
            except Exception as e:
                print(f"Error al copiar {image_path}: {e}")

# Ejemplo de uso
images_base_path = "data/trap-colour-insects-dataset"  # Ruta a las imágenes originales
output_folder = "data/TMP_trap-colour-insects-dataset-yolo"  # Ruta de la carpeta de salida
copy_images_to_temp_folder("data/trap-colour-insects-dataset/train.txt", images_base_path, os.path.join(output_folder, 'train'))
copy_images_to_temp_folder("data/trap-colour-insects-dataset/validation.txt", images_base_path, os.path.join(output_folder, 'val'))

# Load a model
model = YOLO("results/yolov8_trap-colour-insects/yolov8n-cls.pt")
# Train the model
model = model.train(data=output_folder, epochs=5)
# Export the model
model.export(format="edgetpu")


# Borrar la carpeta de salida
try:
    shutil.rmtree(output_folder)
    print(f"La carpeta '{output_folder}' ha sido eliminada.")
except Exception as e:
    print(f"Error al intentar eliminar la carpeta '{output_folder}': {e}")