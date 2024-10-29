import os
import json

def generate_labels_txt(parent_folder):
    # Obtener todas las subcarpetas en el directorio especificado
    labels = [name for name in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, name))]
    
    # Convertir la lista de labels al formato solicitado
    labels_str = json.dumps(labels)
    
    # Escribir las etiquetas en el archivo labels.txt
    with open('labels.txt', 'w') as f:
        f.write(labels_str)
    
    print("Archivo labels.txt generado con éxito.")

# Especifica la ruta de la carpeta principal
parent_folder_path = 'data/imagenet_val'
generate_labels_txt(parent_folder_path)