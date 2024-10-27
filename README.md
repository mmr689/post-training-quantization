<p align="center">
  <img src="assets/final_banner.png" alt="Banner" width="100%">
</p>

The aim of this repository is to learn and understand how [Tensorflow post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) works.

## done

- Genero modelos cuantis de clasificación de TF (tambien h5, pb y tflite).
- Los ejecuto en server y todo correcto con TF.
- Los ejecuto en RPi y todo correcto con TFLite-runtime.
+ Los ejecuto en RPi+EdgeTPU y todo correcto. ESTO ES RARO, TODOS LOS MODELOS SE PUEDEN EJECUTAR EN LA EDGE Y NO DEBERÍA.
   - En tiempos vemos que cargar le cuesta más con la edge y eso es coherente pero la inferencia creo la hace en la CPU.
+ Probar en la coral dev board para corroborar.
- Ya he hecho que only integer funcione. Parece que si tienes FP32 se va a la CPU pero si trabajas en INT8 funciona.
- he podido ejecutar el tflitefp32 de yolo11n y con el usb el edgetpu (baja mucho la precisión)

- Ahora tocaría probar a usar dataset personalizado.


# ToDo list

<table border="1" cellpadding="10">
    <thead>
        <tr>
            <th rowspan="2">Tarea</th>
            <th colspan="2" style="text-align: center;">ImageNet</th>
            <th colspan="2" style="text-align: center;">Colour-insects</th>
        </tr>
        <tr>
            <th style="text-align: center;">MobileNet v2</th>
            <th style="text-align: center;">YOLO-CLS v8n</th>
            <th style="text-align: center;">MobileNet v2</th>
            <th style="text-align: center;">YOLO v8n</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Cuantizar</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">✔️</td>
            <td style="text-align: center;"></td>
        </tr>
        <tr>
            <td>Test general</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">✔️</td>
            <td style="text-align: center;"></td>
        </tr>
        <tr>
            <td>Test edge</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">✔️</td>
            <td style="text-align: center;"></td>
        </tr>
        <tr>
            <td>Métricas modelo</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">✔️</td>
            <td style="text-align: center;"></td>
        </tr>
        <tr>
            <td>Medir tiempo carga modelo</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">✔️</td>
            <td style="text-align: center;"></td>
        </tr>
        <tr>
            <td>Medir tiempo inferencia (primera imagen)</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;">✔️</td>
            <td style="text-align: center;"></td>
        </tr>
        <tr>
            <td>Medir CPU y memoria</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
        </tr>
        <tr>
            <td>Medir consumos</td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
            <td style="text-align: center;"></td>
        </tr>
    </tbody>
</table>

- Dockerizar proyecto (quizás puedo usar el de ultralytics, incluso puede tener el propio dataset dento)
- Averiguar que es cada modelo de Tensorflow
- Averiguar que es cada modelo Ultralytics

---

- `model_quantizer_run_example.py` genera modelos normal, tflite i tflite cuantizados de los modelos clasificadores de imagenet.
- En `examples` hay un código para cada tipo de cuantificación por separado de la manera más raw posible tanto de creaciómo ejecución.
- `model_quantizer_inference_example.py` Me parece que pretende hacer lo mismo que run `run_inference_tflite.py`



Por lo tanto solo funciona con la versión dimanica que debe ser FP32. Explorar cómo ejecutar el resto

The table below shows the tested models.

<table>
   <tr>
      <th>Model</th>
      <th>Purpose</th>
      <th>Source</th>
      <th>Quantification</th>
      <th>EdgeTPU</th>
   </tr>
   <tr>
      <td>MobileNet</td>
      <td>Classification</td>
      <td>Tensorflow</td>
      <td>✔️</td>
      <td>❔</td>
   </tr>
   <tr>
      <td>MobileNetV2</td>
      <td>Classification</td>
      <td>Tensorflow</td>
      <td>✔️</td>
      <td>❔</td>
   </tr>
   <tr>
      <td>VGG16</td>
      <td>Classification</td>
      <td>Tensorflow</td>
      <td>✔️</td>
      <td>❔</td>
   </tr>
</table>

Interesante probar con
- ultralytics
- torchvision
- detectores de objetos...

```bash
sudo docker run -it --ipc=host --gpus all ultralytics/ultralytics:latest

sudo docker run -it --ipc=host --gpus all \
-v /home/qcienmed/mmr689/quantization/results/YOLOv8n:/workspace \
ultralytics/ultralytics:latest

cd /workspace
python yolov8-cls.py
```


## Files

- model_quantizer_run_example.py: Ejemplo de creación de los diferentes tipos de cuantificación con tensorflow con ua clase que lo gestione
- Ejemplos particulares from scratch
