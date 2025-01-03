from ultralytics import RTDETR

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11m.yaml").load("yolo11m.pt")  # build from YAML and transfer weights
model = RTDETR("rtdetr-l.pt")

# Train the model
results = model.train(data="cag/cfg/datasets/sar_aircraft.yaml", epochs=300, device=1, imgsz=1500, batch=4)
