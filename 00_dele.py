import os
import random
import shutil

def select_random_images(source_folder, destination_folder, num_images=50):
    # Crea la carpeta de destino si no existe
    os.makedirs(destination_folder, exist_ok=True)

    # Recorre todas las subcarpetas en la carpeta fuente
    for label in os.listdir(source_folder):
        label_path = os.path.join(source_folder, label)

        # Verifica si es un directorio
        if os.path.isdir(label_path):
            # Obtiene todas las imágenes en la subcarpeta
            images = [img for img in os.listdir(label_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Selecciona aleatoriamente 50 imágenes o menos si hay menos de 50
            selected_images = random.sample(images, min(num_images, len(images)))

            # Crea la carpeta de destino para esta etiqueta
            label_dest_path = os.path.join(destination_folder, label)
            os.makedirs(label_dest_path, exist_ok=True)

            # Copia las imágenes seleccionadas a la carpeta de destino
            for img in selected_images:
                src_img_path = os.path.join(label_path, img)
                dest_img_path = os.path.join(label_dest_path, img)
                shutil.copy(src_img_path, dest_img_path)

# Especifica la ruta a la carpeta imagenet1k y la carpeta de destino
source_folder = "data/imagenet1k"  # Cambia esto a la ruta de tu carpeta
destination_folder = "data/imagenet_val"  # Cambia esto a la ruta donde quieras guardar las imágenes seleccionadas

select_random_images(source_folder, destination_folder, num_images=5)
