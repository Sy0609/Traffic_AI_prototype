
import cv2
import numpy as np

from opencv_basics import preprocess_frame, save_outputs


def average_line(line_list, height):
    if not line_list:
        return None

    points = np.array(line_list, dtype=np.float64).reshape(-1, 2)
    xs = points[:, 0]
    ys = points[:, 1]

    # Fit duong thang theo dang x = a*y + b
    a, b = np.polyfit(ys, xs, 1)

    y_bottom = height - 1
    y_top = int(height * 0.60)

    x_bottom = int(a * y_bottom + b)
    x_top = int(a * y_top + b)

    return x_bottom, y_bottom, x_top, y_top


def detect_lanes(frame, edges):
    height, width = frame.shape[:2]

    # 1. Tao vung ROI
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (int(width * 0.1), height - 1),
        (int(width * 0.45), int(height * 0.55)),
        (int(width * 0.55), int(height * 0.55)),
        (int(width * 0.9), height - 1)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)

    # 2. Tim cac doan thang bang Hough Transform
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=40,
        maxLineGap=100
    )

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(-1)[:4]

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.4:
                continue

            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    # 3. Ve duong dai dai dien cho lan duong
    result = frame.copy()

    left_line = average_line(left_lines, height)
    right_line = average_line(right_lines, height)

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 5)

    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return roi, result


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

        gray, blur, edges = preprocess_frame(frame)
        roi, result = detect_lanes(frame, edges)

        cv2.imshow("Original Video", frame)
        cv2.imshow("Gray Image", gray)
        cv2.imshow("Canny Edges", edges)
        cv2.imshow("ROI Edges", roi)
        cv2.imshow("Lane Lines", result)

        if not saved_images:
            save_outputs(frame, gray, edges, roi, result)
            saved_images = True

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Hoan thanh!")
