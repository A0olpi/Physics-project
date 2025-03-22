import cv2
from ultralytics import YOLO

# Load the YOLOv8 Large model
model = YOLO("yolov8l.pt")  # Use "yolov8l.pt" for Large or "yolov8x.pt" for Extra Large

# Open the webcam
cap = cv2.VideoCapture(0)

# Variable to toggle detection on/off
detection_active = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference if detection is active
    if detection_active:
        results = model(frame)  # Perform object detection
        annotated_frame = results[0].plot()  # Get the annotated frame with bounding boxes
    else:
        annotated_frame = frame  # Display the original frame without detection

    # Display the frame
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Toggle detection on/off with 'd' key
    key = cv2.waitKey(1)
    if key & 0xFF == ord('d'):
        detection_active = not detection_active
        print(f"Detection {'active' if detection_active else 'inactive'}")

    # Exit on 'q' key
    if key & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()