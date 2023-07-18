from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="C:/M1/stage/Camera_trap/lynx_entrainement/data_local.yaml", epochs=50)  # train the model
