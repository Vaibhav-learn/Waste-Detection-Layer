from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source= "103.webp",show = True , save = True)