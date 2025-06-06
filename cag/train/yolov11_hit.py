from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11m.yaml").load("yolo11m.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="cag/cfg/datasets/hit.yaml", epochs=300, imgsz=640, device="0,1")
