
from ultralytics import YOLO


class TrafficSignDetector:
    def __init__(self, model_path="runs/detect/runs/traffic_signs/yolo26n_gtsdb/weights/best.pt", conf=0.25):
        """
        Khởi tạo bộ nhận diện.
        """
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """
        Input:
            frame: Ảnh BGR từ OpenCV.

        Output:
            annotated_frame: Ảnh đã vẽ kết quả.
            detections: Danh sách thông tin đối tượng.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=640,
            device="cpu",
            verbose=False
        )

        result = results[0]
        annotated_frame = result.plot()

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": confidence,
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ]
                })

        return annotated_frame, detections
