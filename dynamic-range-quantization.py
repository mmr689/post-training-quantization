"""
Dynamic range quantization
https://www.tensorflow.org/lite/performance/post_training_quantization#dynamic_range_quantization
"""

import tensorflow as tf

saved_model_dir = '/home/qcienmed/mmr689/quantization/PIUsatge/'

# Cargar el modelo y listar las firmas disponibles
loaded_model = tf.saved_model.load(saved_model_dir)
print('Signatures: ',list(loaded_model.signatures.keys()))

# Acceder a una firma específica y ver detalles
if 'serving_default' in loaded_model.signatures:
    infer = loaded_model.signatures['serving_default']
    print("Detalles de la firma 'serving_default':")
    print(" · Entradas:", infer.structured_input_signature)
    print(" · Salidas:", infer.structured_outputs)


# Configurar el convertidor
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convertir el modelo
tflite_quant_model = converter.convert()

# Guardar el modelo TFLite
with open(saved_model_dir+'modelo_cuantificado.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print('E N D')