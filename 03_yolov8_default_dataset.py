"""
Código para emplear el modelo por default de ultralytics.
"""

from ultralytics import YOLO

# Load pretrained model
model = YOLO("results/yolov8_imagenet-default/yolov8n-cls.pt")

# Export the model
model.export(format="edgetpu")