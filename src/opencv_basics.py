
import cv2
import os


def preprocess_frame(frame):
    """Chuyen frame sang anh xam, lam mo va phat hien canh."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    return gray, blur, edges


def save_outputs(frame, gray, edges, roi, result):
    """Luu cac anh ket qua vao thu muc outputs."""
    os.makedirs("outputs", exist_ok=True)

    cv2.imwrite("outputs/original.jpg", frame)
    cv2.imwrite("outputs/gray.jpg", gray)
    cv2.imwrite("outputs/edges.jpg", edges)
    cv2.imwrite("outputs/roi.jpg", roi)
    cv2.imwrite("outputs/lane_lines.jpg", result)

    print("Da luu 5 anh vao thu muc outputs!")
