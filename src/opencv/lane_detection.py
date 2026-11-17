
import cv2
import numpy as np

from .opencv_basics import preprocess_frame, save_outputs



def group_lane_lines(lines, width, height):
    """
    Phân loại các đoạn thẳng Hough thành làn trái và làn phải.
    Lọc các đoạn thẳng không phù hợp trước khi phân loại.
    """

    left_lines = []
    right_lines = []

    if lines is None:
        return left_lines, right_lines

    center_x = width // 2

    # Độ dài tối thiểu của một đoạn thẳng
    min_length = max(30, int(height * 0.05))

    for line in lines:
        x1, y1, x2, y2 = line.reshape(-1)[:4]

        dx = x2 - x1
        dy = y2 - y1

        # 1. Loại đoạn thẳng quá ngắn theo trục Y
        if abs(dy) < 10:
            continue

        # 2. Tính độ dài đoạn thẳng
        length = np.sqrt(dx ** 2 + dy ** 2)

        if length < min_length:
            continue

        # 3. Tính độ dốc theo dạng x = a*y + b
        slope = dx / dy

        # Loại đoạn quá thoải hoặc quá dốc
        if abs(slope) < 0.1 or abs(slope) > 1.5:
            continue

        # 4. Tính vị trí trung bình của đoạn thẳng
        center_line_x = (x1 + x2) / 2

        # 5. Phân loại làn trái và làn phải
        if slope < 0 and center_line_x < center_x:
            left_lines.append((x1, y1, x2, y2))

        elif slope > 0 and center_line_x > center_x:
            right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines



def average_line(line_list, height, width):
    """
    Gộp nhiều đoạn thẳng thành một đường đại diện.
    Sử dụng trung bình có trọng số theo độ dài đoạn thẳng.
    """

    if not line_list:
        return None

    slopes = []
    intercepts = []
    weights = []

    for x1, y1, x2, y2 in line_list:

        dy = y2 - y1

        if abs(dy) < 10:
            continue

        # Phương trình: x = a*y + b
        slope = (x2 - x1) / dy
        intercept = x1 - slope * y1

        length = np.sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2
        )

        slopes.append(slope)
        intercepts.append(intercept)
        weights.append(length)

    if not slopes:
        return None

    average_slope = np.average(slopes, weights=weights)
    average_intercept = np.average(intercepts, weights=weights)

    # Xác định hai điểm đầu và cuối của đường đại diện
    y_bottom = height - 1
    y_top = int(height * 0.60)

    x_bottom = int(average_slope * y_bottom + average_intercept)
    x_top = int(average_slope * y_top + average_intercept)

    # Giới hạn tọa độ trong phạm vi ảnh
    x_bottom = int(np.clip(x_bottom, 0, width - 1))
    x_top = int(np.clip(x_top, 0, width - 1))

    return x_bottom, y_bottom, x_top, y_top


def detect_lanes(frame, edges):
    """
    Phát hiện làn trái và làn phải từ ảnh Canny.
    """

    height, width = frame.shape[:2]

    # 1. Tạo vùng ROI
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (int(width * 0.1), height - 1),
        (int(width * 0.45), int(height * 0.55)),
        (int(width * 0.55), int(height * 0.55)),
        (int(width * 0.9), height - 1)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    roi = cv2.bitwise_and(edges, mask)

    # 2. Phát hiện các đoạn thẳng bằng Hough Transform
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=40,
        maxLineGap=100
    )

    # 3. Phân loại đoạn thẳng
    left_lines, right_lines = group_lane_lines(
        lines, width, height
    )

    # 4. Gộp các đoạn thẳng thành hai đường đại diện
    left_lane = average_line(left_lines, height, width)
    right_lane = average_line(right_lines, height, width)

    # 5. Vẽ lên ảnh gốc
    result = frame.copy()

    if left_lane is not None:
        x1, y1, x2, y2 = left_lane

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            5
        )

    if right_lane is not None:
        x1, y1, x2, y2 = right_lane

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            5
        )

    # 6. Xác định trạng thái phát hiện làn
    if left_lane is not None and right_lane is not None:
        status = "Both lanes detected"

    elif left_lane is not None:
        status = "Missing right lane"

    elif right_lane is not None:
        status = "Missing left lane"

    else:
        status = "No lanes detected"

    cv2.putText(
        result,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # Thông tin phục vụ kiểm tra
    lane_info = {
        "left_lane": left_lane,
        "right_lane": right_lane,
        "left_segments": len(left_lines),
        "right_segments": len(right_lines),
        "status": status
    }

    return roi, result, lane_info


def run_lane_detection(video_path="videos/road.mp4"):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Khong mo duoc video:", video_path)
        return

    saved_images = False

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Tiền xử lý ảnh
        gray, blur, edges = preprocess_frame(frame)

        # Phát hiện làn đường
        roi, result, lane_info = detect_lanes(frame, edges)

        # Hiển thị kết quả
        cv2.imshow("Original Video", frame)
        cv2.imshow("Gray Image", gray)
        cv2.imshow("Gaussian Blur", blur)
        cv2.imshow("Canny Edges", edges)
        cv2.imshow("ROI Edges", roi)
        cv2.imshow("Lane Detection", result)

        # Lưu kết quả frame đầu tiên
        if not saved_images:
            save_outputs(frame, gray, edges, roi, result)
            saved_images = True

        # Nhấn Q để thoát
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Hoan thanh!")


if __name__ == "__main__":
    run_lane_detection()
