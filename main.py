import cv2
import time

# 1. Mở webcam (Số 0). Nếu em có file video, đổi 0 thành tên file, ví dụ "road.mp4"
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Lỗi: Không mở được camera hoặc video.")
    exit()

prev_time = time.perf_counter()

# 2. Vòng lặp đọc từng khung hình (frame)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Đã hết video hoặc mất kết nối camera.")
        break

    # Tính FPS
    curr_time = time.perf_counter()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Ghi FPS lên góc trái khung hình
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 3. Hiển thị cửa sổ
    cv2.imshow("TV3 - He Thong Nhan Dien Bien Bao & Lan Duong", frame)

    # Bấm phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera khi thoát
cap.release()
cv2.destroyAllWindows()