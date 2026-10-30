
import cv2

from .opencv_basics import preprocess_frame


def process_video(video_path="videos/road.mp4"):
    # 1. Mo video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Khong the mo video:", video_path)
        return

    # 2. Lay thong tin video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Mo video thanh cong!")
    print("Chieu rong:", width)
    print("Chieu cao:", height)
    print("FPS:", fps)
    print("Tong so frame:", total_frames)

    # 3. Doc va xu ly tung frame
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Da het video!")
            break

        gray, blur, edges = preprocess_frame(frame)

        # 4. Hien thi ket qua
        cv2.imshow("Original Video", frame)
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Gaussian Blur", blur)
        cv2.imshow("Canny Edges", edges)

        # 5. Nhan q de thoat
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    # 6. Giai phong tai nguyen
    cap.release()
    cv2.destroyAllWindows()

    print("Hoan thanh xu ly video!")


if __name__ == "__main__":
    process_video()
