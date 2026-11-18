import cv2
import time

 HEAD
# ==========================================
# KHU VỰC DÀNH CHO TV1 (AI - Nhận diện biển báo)
# ==========================================
def detect_traffic_signs(frame):
    # Dữ liệu giả lập: Vẽ khung đỏ góc trái để test giao diện
    cv2.rectangle(frame, (50, 50), (200, 200), (0, 0, 255), 2)
    cv2.putText(frame, "Bien Bao (Gia)", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame

# ==========================================
# KHU VỰC DÀNH CHO TV2 (OpenCV - Làn đường)
# ==========================================
def detect_lanes(frame):
    # Dữ liệu giả lập: Vẽ 2 đường chéo màu xanh dương làm làn đường
    height, width = frame.shape[:2]
    cv2.line(frame, (int(width * 0.2), height), (int(width * 0.4), int(height * 0.6)), (255, 0, 0), 3)
    cv2.line(frame, (int(width * 0.8), height), (int(width * 0.6), int(height * 0.6)), (255, 0, 0), 3)
    return frame

# ==========================================
# KHU VỰC DÀNH CHO TV3 (Tích hợp & Hiển thị)
# ==========================================
# Dùng file video có sẵn trong thư mục videos hoặc dùng webcam (0)
VIDEO_SOURCE = "videos/road.mp4" # Hoặc đổi thành 0 nếu dùng webcam

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Lỗi: Không mở được video hoặc camera. Kiểm tra lại đường dẫn!")

# 1. Mở webcam (Số 0). Nếu em có file video, đổi 0 thành tên file, ví dụ "road.mp4"
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Lỗi: Không mở được camera hoặc video.")
 b07ef179a27482644094c5dc076125aaee5a1ab9
    exit()

prev_time = time.perf_counter()

 HEAD
while True:
    ret, frame = cap.read()
    if not ret:
        print("Đã hết video hoặc mất kết nối.")
        break

    # 1. Gọi hàm nhận diện của TV1
    frame = detect_traffic_signs(frame)

    # 2. Gọi hàm nhận diện của TV2
    frame = detect_lanes(frame)

    # 3. TV3 tính toán FPS sau khi qua các lớp xử lý

# 2. Vòng lặp đọc từng khung hình (frame)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Đã hết video hoặc mất kết nối camera.")
        break
    # Tính FPS
 b07ef179a27482644094c5dc076125aaee5a1ab9
    curr_time = time.perf_counter()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

 HEAD
    # 4. In FPS và hiển thị kết quả cuối cùng
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("TV3 - He Thong Tich Hop AI & OpenCV", frame)

    # Bấm phím 'q' để thoát chương trình
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Ghi FPS lên góc trái khung hình
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 3. Hiển thị cửa sổ
    cv2.imshow("TV3 - He Thong Nhan Dien Bien Bao & Lan Duong", frame)

    # Bấm phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera khi thoát
>>>>>>> b07ef179a27482644094c5dc076125aaee5a1ab9
cap.release()
cv2.destroyAllWindows()