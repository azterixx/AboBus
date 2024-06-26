from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# res = model.track(0, show=True, classes=[0])

res = model.predict(0, show=True)
