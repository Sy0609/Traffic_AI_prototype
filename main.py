from collections import Counter, deque
from math import isfinite
from time import perf_counter, sleep

import cv2

# 1. Import module nhận diện biển báo của TV1
from detect_signs import TrafficSignDetector

# 2. Import module xử lý ảnh và làn đường của TV2
from src.opencv.opencv_basics import preprocess_frame
from src.opencv.lane_detection import detect_lanes


# --- CẤU HÌNH THÔNG SỐ TV3 ---
VIDEO_PATH = "videos/road.mp4"  # Đường dẫn video test chuẩn của nhóm
CONFIDENCE = 0.25
INFERENCE_INTERVAL = 2          # Bỏ qua một số frame để tăng tốc độ xử lý YOLO
VOTE_WINDOW = 3
VOTES_REQUIRED = 2
MATCH_IOU = 0.30
SIGN_DISPLAY_SECONDS = 1.0
WINDOW_NAME = "TV3 - He Thong Tich Hop AI Bien Bao & Lan Duong"


def iou(box_a, box_b):
    """Tính tỷ lệ giao trên hợp của hai khung."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    overlap_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = overlap_width * overlap_height
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection

    return intersection / union if union > 0 else 0.0


def update_tracks(tracks, detections):
    """Ghép khung theo vị trí và cập nhật các phiếu của từng biển."""
    candidates = []

    for track_index, track in enumerate(tracks):
        track["labels"].append(None)
        for detection_index, detection in enumerate(detections):
            overlap = iou(track["bbox"], detection["bbox"])
            if overlap >= MATCH_IOU:
                candidates.append((overlap, track_index, detection_index))

    used_tracks = set()
    matched = {}

    for _, track_index, detection_index in sorted(candidates, reverse=True):
        if track_index in used_tracks or detection_index in matched:
            continue
        used_tracks.add(track_index)
        matched[detection_index] = tracks[track_index]

    visible_tracks = []

    for index, detection in enumerate(detections):
        track = matched.get(index)
        if track is None:
            track = {"labels": deque([None] * VOTE_WINDOW, maxlen=VOTE_WINDOW)}
            tracks.append(track)

        track["bbox"] = detection["bbox"]
        track["labels"][-1] = detection["class_name"]
        visible_tracks.append(track)

    tracks[:] = [track for track in tracks if any(track["labels"])]
    return visible_tracks


def draw_tracks(frame, visible_tracks):
    """Vẽ bounding box và nhãn biển báo lên khung hình."""
    for track in visible_tracks:
        votes = Counter(label for label in track["labels"] if label is not None)
        if not votes:
            continue
        label, count = votes.most_common(1)[0]

        if count >= VOTES_REQUIRED and (
            "confirmed_label" not in track or count == VOTE_WINDOW
        ):
            track["confirmed_label"] = label

        confirmed_label = track.get("confirmed_label")
        text = confirmed_label if confirmed_label is not None else "Dang xac nhan"
        color = (0, 220, 0) if confirmed_label is not None else (150, 150, 150)

        x1, y1, x2, y2 = track["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, text, (x1, max(22, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
        )


def main():
    detector = TrafficSignDetector(conf=CONFIDENCE)
    cap = cv2.VideoCapture(VIDEO_PATH)

    try:
        if not cap.isOpened():
            raise SystemExit(f"Khong mo duoc video tai: {VIDEO_PATH}")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
        frame_period = 1.0 / source_fps

        tracks = []
        visible_tracks = []
        last_sign_text = ""
        show_sign_until = 0.0

        frame_index = 0
        next_display_at = None
        fps_start = perf_counter()
        displayed_frames = 0
        inference_frames = 0
        fps_text = "FPS hien thi: -- | YOLO: --"

        print("Đang khởi động hệ thống tích hợp TV3 (Biển báo YOLO + Làn đường OpenCV)... Nhấn 'q' để thoát.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Đã phát xong video.")
                break

            # 1. Xử lý tiền xử lý ảnh và phát hiện làn đường từ TV2
            gray, blur, edges = preprocess_frame(frame)
            _, lane_frame = detect_lanes(frame, edges)  # lane_frame đã được vẽ vạch làn đường xanh

            # 2. Xử lý nhận diện biển báo YOLO từ TV1 theo chu kỳ frame
            if frame_index % INFERENCE_INTERVAL == 0:
                _, detections = detector.detect(frame)
                visible_tracks = update_tracks(tracks, detections)
                inference_frames += 1

            # 3. Vẽ bounding box biển báo đè lên khung hình làn đường
            draw_tracks(lane_frame, visible_tracks)
            frame_index += 1

            # Đồng bộ tốc độ khung hình và tính FPS
            now = perf_counter()
            if next_display_at is None:
                next_display_at = now
                fps_start = now

            delay = next_display_at - now
            if delay > 0:
                sleep(delay)
            next_display_at += frame_period

            displayed_frames += 1
            now = perf_counter()
            elapsed = now - fps_start

            if elapsed >= 1.0:
                fps_text = (
                    f"FPS hien thi: {displayed_frames / elapsed:.1f} | "
                    f"YOLO: {inference_frames / elapsed:.1f}"
                )
                fps_start = now
                displayed_frames = 0
                inference_frames = 0

            # Hiển thị thông số FPS góc trên trái
            cv2.rectangle(lane_frame, (10, 10), (460, 52), (0, 0, 0), -1)
            cv2.putText(
                lane_frame, fps_text, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            )

            confirmed_labels = sorted({
                track["confirmed_label"]
                for track in visible_tracks
                if "confirmed_label" in track
            })

            if confirmed_labels:
                last_sign_text = ", ".join(confirmed_labels)
                show_sign_until = now + SIGN_DISPLAY_SECONDS

            if now < show_sign_until:
                cv2.rectangle(lane_frame, (10, 58), (460, 98), (0, 0, 0), -1)
                cv2.putText(
                    lane_frame, f"Vua nhan dien: {last_sign_text}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                )

            # Hiển thị cửa sổ tổng hợp kết quả (vừa có làn đường, vừa có biển báo, vừa có FPS)
            cv2.imshow(WINDOW_NAME, lane_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()