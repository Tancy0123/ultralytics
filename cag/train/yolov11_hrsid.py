from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11m.yaml").load("./runs/detect/train33/weights/best.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="cag/cfg/datasets/HRSID.yaml", epochs=2, imgsz=800, device="0,1")
