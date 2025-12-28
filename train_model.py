from roboflow import Roboflow
from ultralytics import YOLO
import os

if __name__ == '__main__':

# download data
    rf = Roboflow(api_key="EiSbTr5gAjGrRXm3UN8C")
project = rf.workspace("object-detection-oitsh").project("lemon-harvesting-bpkrb")
version = project.version(1)
dataset = version.download("yolov11")

# train model
model = YOLO('yolo11n.pt')
results = model.train(
    data=os.path.join(dataset.location, 'data.yaml'),
    epochs = 50,
    imgsz = 640,
    workers = 0,
    device = 'cpu'
)

print("Training Finished!!!")
print(f"Your brain is saved at: runs/detect/train/weights/best.pt")
