from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weights/yolov8x.pt')
results = model("Images/1.jpg")

# Plot the image with bounding boxes
annotated_frame = results[0].plot()

cv2.imshow("Results", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
