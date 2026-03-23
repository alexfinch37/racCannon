from ultralytics import YOLO

model = YOLO("my_newmodel1.pt")

results = model.predict(source="testSq.jpeg", save=True, show=True)
