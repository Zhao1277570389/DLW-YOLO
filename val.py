


#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model

metrics = model.val(data="coco128.yaml")  # evaluate model performance on the validation set
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 