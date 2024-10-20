from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo11n-cls.pt")

# Export the model
model.export(format="edgetpu")