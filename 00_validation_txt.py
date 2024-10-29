import os

def generate_validation_txt(parent_folder):
    with open('validation.txt', 'w') as f:
        # Recorrer cada subcarpeta dentro de la carpeta principal
        for subfolder in os.listdir(parent_folder):
            subfolder_path = os.path.join(parent_folder, subfolder)
            # Verificar si el elemento es una subcarpeta
            if os.path.isdir(subfolder_path):
                # Recorrer cada archivo dentro de la subcarpeta
                for filename in os.listdir(subfolder_path):
                    # Filtrar solo archivos de imagen (ajusta las extensiones según tus necesidades)
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Escribir en el archivo el formato "subcarpeta/nombre_imagen.ext"
                        f.write(f"{subfolder}/{filename}\n")
    
    print("Archivo validation.txt generado con éxito.")

# Especifica la ruta de la carpeta principal
parent_folder_path = 'data/imagenet_val'
generate_validation_txt(parent_folder_path)
