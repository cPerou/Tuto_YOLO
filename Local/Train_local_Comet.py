from ultralytics import YOLO

# Utiliser Comet pour suivre l'entrainement et comparer avec d'autres modèles
from comet_ml import Experiment

# Synchroniser mon compte Comet
experiment = Experiment(
  api_key = "lkOAeZvucVeCbUPfsPeJlRqkf",
  project_name = "lynx",
  workspace="cperou"
)

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="C:/M1/stage/Camera_trap/lynx_entrainement/dataLocal.yaml", epochs=100)  # train the model