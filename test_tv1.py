
import cv2
from detect_signs import TrafficSignDetector

# Khởi tạo mô hình một lần
detector = TrafficSignDetector(
    model_path="yolo26n.pt",
    conf=0.25
)

# Mở video kiểm thử
cap = cv2.VideoCapture("video/test.mp4")

if not cap.isOpened():
    print("Lỗi: Không mở được video.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Đã xử lý hết video.")
        break

    # Nhận diện và vẽ kết quả
    annotated_frame, detections = detector.detect(frame)
    for detection in detections:
     print(detection)
    # Hiển thị kết quả
    cv2.imshow("TV1 - Detection Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
