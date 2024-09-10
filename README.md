<p align="center">
  <img src="assets/final_banner.png" alt="Banner" width="100%">
</p>

The aim of this repository is to learn and understand how [Tensorflow post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) works.

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