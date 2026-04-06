from ultralytics import YOLO
import cv2
import numpy as np
import time

FONT_SIZE = 7
FONT_THICKNESS = 4

def make_alert_frame():
    alert_img = np.zeros((200, 400, 3), dtype=np.uint8)
    alert_img[:] = (30, 30, 220)
    cv2.putText(alert_img, "PERSON DETECTED",
                (30, 80), cv2.FONT_ITALIC, 0.9, (255, 255, 255), 2)
    cv2.putText(alert_img, "Press D to dismiss",
                (90, 140), cv2.FONT_ITALIC, 0.6, (200, 200, 200), 1)
    return alert_img


model1 = YOLO('yolov8n.pt')

DISABLED_CLASSES = ['cat', 'bear', 'bird']

last_alert_time = 0
ALERT_COOLDOWN = 5
alert_active = False
alert_start_time = 0
ALERT_DURATION = 5

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results1 = model1(frame, conf=0.8, verbose=False)

    boxes = []
    person_detected = False

    for box in results1[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = model1.names[int(box.cls[0])]

        if cls in DISABLED_CLASSES:
            continue
        if cls == 'person':
            person_detected = True

        boxes.append({
            'box': [x1, y1, x2, y2],
            'conf': conf,
            'label': cls,
            'color': (0, 0, 255)
        })

    current_time = time.time()

    if person_detected and (current_time - last_alert_time) > ALERT_COOLDOWN:
        alert_active = True
        alert_start_time = current_time
        last_alert_time = current_time

    if alert_active and (current_time - alert_start_time) > ALERT_DURATION:
        alert_active = False
        cv2.destroyWindow("PERSON DETECTED")

    if alert_active:
        cv2.imshow("PERSON DETECTED", make_alert_frame())

    annotated_frame = frame.copy()
    for detection in boxes:
        x1, y1, x2, y2 = detection['box']
        label = f"{detection['label']} {detection['conf']:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 255), FONT_THICKNESS)

    cv2.putText(annotated_frame, f"Detections: {len(boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Detection', annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('d') and alert_active:
        alert_active = False
        cv2.destroyWindow("PERSON DETECTED")

cap.release()
cv2.destroyAllWindows()