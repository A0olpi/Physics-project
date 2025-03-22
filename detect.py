import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression

# Load YOLOv5 model
model = attempt_load('yolov5l.pt')

# Open webcam
cap = cv2.VideoCapture(0)

# Toggle variable
detection_active = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Toggle detection on/off with the 'd' key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):  # Press 'd' to toggle detection
        detection_active = not detection_active
        print(f"Detection {'active' if detection_active else 'inactive'}")

    if detection_active:
        # Convert frame to tensor
        frame_tensor = torch.from_numpy(frame).float()  # Convert to float tensor
        frame_tensor = frame_tensor.permute(2, 0, 1)  # Change shape from HWC to CHW
        frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension

        # Run inference
        results = model(frame_tensor)
        detections = non_max_suppression(results)

        # Overlay results
        for detection in detections[0]:  # detections is a list of tensors
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Frame', frame)

    # Exit on 'q' key
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()