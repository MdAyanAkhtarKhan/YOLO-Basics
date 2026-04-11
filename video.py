from ultralytics import YOLO
import cv2
import cvzone
import math

# Video path
#cap = cv2.VideoCapture(r"C:\Users\AYAN\PycharmProjects\PythonProject_computer_vision\venv310\Videos\test02.mp4")
cap = cv2.VideoCapture("Videos/test02.mp4")

# Load YOLO model
model = YOLO("yolo-weights/yolov8x.pt")

# COCO Classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Main loop
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (1280, 720))

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            conf = round(float(box.conf[0]), 2)

            if conf < 0.7:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])

            label = classNames[cls] if cls < len(classNames) else "Unknown"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(
                img,
                f'{label} {conf}',
                (max(0, x1), max(35, y1)),
                scale=0.9,
                thickness=1,
                colorR=(0, 0, 255),
            )

    cv2.imshow("Video YOLO", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()