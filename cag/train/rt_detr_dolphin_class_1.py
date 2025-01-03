from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="cag/cfg/datasets/dolphin_class_1.yaml", epochs=300, imgsz=1024)

# # Run inference with the RT-DETR-l model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")
