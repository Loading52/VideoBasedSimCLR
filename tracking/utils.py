import os
import cv2


class VideoInput:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()

    def print_video_info(self):
        """
        打印视频的基本信息，包括分辨率、帧率和总帧数。
        """
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        print("\n=== Video Input Info ===")
        print(f"Resolution: {width}x{height}")
        print(f"Frame Rate: {fps:.2f} FPS")
        print(f"Total Frames: {self.total_frames}")
        print("========================\n")


class ImageSequenceInput:
    def __init__(self, image_folder):
        self.images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")])
        self.image_folder = image_folder
        self.index = 0
        self.total_frames = len(self.images)  # 通过图片数量确定总帧数

    def read_frame(self):
        if self.index >= self.total_frames:
            return None
        img_path = os.path.join(self.image_folder, self.images[self.index])
        frame = cv2.imread(img_path)
        self.index += 1
        return frame

    def release(self):
        pass

    def print_video_info(self):
        """
        打印图像序列的基本信息，包括分辨率和总帧数。
        """
        if self.total_frames == 0:
            print("\n=== Image Sequence Input Info ===")
            print("No images found in the specified folder.")
            print("==================================\n")
            return

        first_image_path = os.path.join(self.image_folder, self.images[0])
        first_image = cv2.imread(first_image_path)
        height, width, _ = first_image.shape

        print("\n=== Image Sequence Input Info ===")
        print(f"Resolution: {width}x{height}")
        print(f"Total Frames: {self.total_frames}")
        print("==============================\n")