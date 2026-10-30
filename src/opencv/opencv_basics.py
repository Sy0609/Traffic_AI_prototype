
import cv2
import os


def preprocess_frame(frame):
    """Chuyen frame sang anh xam, lam mo va phat hien canh."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    return gray, blur, edges


def save_outputs(frame, gray, edges, roi, result):
    """Luu cac anh ket qua vao thu muc outputs/opencv."""
    os.makedirs("outputs/opencv", exist_ok=True)

    cv2.imwrite("outputs/opencv/original.jpg", frame)
    cv2.imwrite("outputs/opencv/gray.jpg", gray)
    cv2.imwrite("outputs/opencv/edges.jpg", edges)
    cv2.imwrite("outputs/opencv/roi.jpg", roi)
    cv2.imwrite("outputs/opencv/lane_lines.jpg", result)

    print("Da luu 5 anh vao outputs/opencv!")
