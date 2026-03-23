from ultralytics import YOLO
import cv2
import numpy as np


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


# Load models
model1 = YOLO('yolo11n.pt')  # Priority model
model2 = YOLO('my_newmodel1.pt')  # Secondary model

# Classes to disable from model1
DISABLED_CLASSES = ['cat']  # Add more classes here: ['cat', 'dog', 'bird']

source = 0
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run both models
    results1 = model1(frame, conf=0.8, verbose=False)
    results2 = model2(frame, conf=0.85, verbose=False)

    # Store model1 detections (these take priority) - FILTER OUT DISABLED CLASSES
    priority_boxes = []
    for box in results1[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = model1.names[int(box.cls[0])]

        # Skip disabled classes
        if cls in DISABLED_CLASSES:
            continue

        priority_boxes.append({
            'box': [x1, y1, x2, y2],
            'conf': conf,
            'label': cls,
            'color': (0, 0, 255)  # Red for model1 (priority)
        })

    # Add model2 detections ONLY if they don't overlap with model1
    secondary_boxes = []
    for box in results2[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = model2.names[int(box.cls[0])]
        new_box = [x1, y1, x2, y2]

        # Check if this overlaps with ANY model1 detection
        overlaps_with_model1 = False
        for priority in priority_boxes:
            if calculate_iou(new_box, priority['box']) > 0.3:  # 30% overlap threshold
                overlaps_with_model1 = True
                break

        # Only add if it doesn't overlap with model1
        if not overlaps_with_model1:
            secondary_boxes.append({
                'box': new_box,
                'conf': conf,
                'label': cls,
                'color': (0, 255, 0)  # Green for model2
            })

    # Combine all boxes (model1 first, then model2)
    all_boxes = priority_boxes + secondary_boxes

    # Draw all boxes
    annotated_frame = frame.copy()
    for detection in all_boxes:
        x1, y1, x2, y2 = detection['box']
        label = f"{detection['label']} {detection['conf']:.2f}"
        color = detection['color']

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display info
    info_text = f"Model1 (Red): {len(priority_boxes)} | Model2 (Green): {len(secondary_boxes)}"
    cv2.putText(annotated_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Combined - Model1 Overrides Model2', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
