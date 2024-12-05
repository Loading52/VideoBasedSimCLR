import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import cv2
import numpy as np
from utils import VideoInput, ImageSequenceInput


class ObjectTracker:
    def __init__(self, max_history=30):
        self.trackers = {}
        self.next_id = 0
        self.max_history = max_history

    def add_tracker(self, initial_position):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        kf.statePre = np.array([initial_position[0], initial_position[1], 0, 0], dtype=np.float32)
        kf.statePost = np.array([initial_position[0], initial_position[1], 0, 0], dtype=np.float32)

        kf.correct(np.array(initial_position, dtype=np.float32))

        self.trackers[self.next_id] = {"kf": kf, "position": initial_position, "history": [initial_position]}
        self.next_id += 1
        return self.next_id - 1

    def update_trackers(self, detections):
        updated_positions = {}

        for tracker_id, tracker_info in list(self.trackers.items()):
            kf = tracker_info["kf"]
            predicted = kf.predict()
            predicted_position = (int(predicted[0]), int(predicted[1]))

            updated_position = None
            for detection in detections:
                if self.match(predicted_position, detection):
                    kf.correct(np.array(detection, dtype=np.float32))
                    updated_position = detection
                    detections.remove(detection)
                    break

            if updated_position is None:
                updated_position = predicted_position

            tracker_info["position"] = updated_position
            tracker_info["history"].append(updated_position)

            if len(tracker_info["history"]) > self.max_history:
                tracker_info["history"].pop(0)

            updated_positions[tracker_id] = updated_position

        for detection in detections:
            tracker_id = self.add_tracker(detection)
            updated_positions[tracker_id] = detection

        return updated_positions

    def match(self, predicted, detection, threshold=50):
        distance = np.linalg.norm(np.array(predicted) - np.array(detection))
        return distance < threshold

    def get_tracks(self):
        return {tracker_id: tracker_info["history"] for tracker_id, tracker_info in self.trackers.items()}

def save_cropped_object(frame, position, size, tracker_id, frame_number, output_dir="output"):
    """
    裁剪物体并保存为图片。
    参数:
    - frame: 当前帧图像
    - position: 物体中心位置 (x, y)
    - size: 物体大小 (w, h)
    - tracker_id: 物体对应的跟踪器 ID
    - frame_number: 当前帧号
    - output_dir: 保存输出的根目录
    """
    x, y = position
    w, h = size
    crop_size = max(w, h)  # 保证裁剪框长宽比为1:1
    x1 = max(0, x - crop_size // 2)
    y1 = max(0, y - crop_size // 2)
    x2 = min(frame.shape[1], x1 + crop_size)
    y2 = min(frame.shape[0], y1 + crop_size)

    # 裁剪并缩放到200x200
    cropped_img = frame[y1:y2, x1:x2]
    resized_img = cv2.resize(cropped_img, (200, 200))

    # 创建保存目录
    object_dir = os.path.join(output_dir, f"object_{tracker_id}")
    os.makedirs(object_dir, exist_ok=True)

    # 保存裁剪图片
    output_path = os.path.join(object_dir, f"frame_{frame_number}.jpg")
    cv2.imwrite(output_path, resized_img)
    print(f"Saved cropped object to {output_path}")

def run_tracking(input_source, output_file="tracking_results.txt", output_dir="output/video_5"):
    frame_number = 0
    ret = input_source.read_frame()
    if ret is None:
        print("No frames to process.")
        return

    prev_gray = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    motion_threshold = 2.0
    min_contour_area = 500
    tracker = ObjectTracker()

    print(f"Processing started. Saving results to '{output_file}'")
    total_frames = input_source.total_frames  # 使用统一的接口

    with open(output_file, "w") as f:
        while True:
            frame = input_source.read_frame()
            if frame is None:
                break
            frame_number += 1

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **flow_params)

            avg_flow_x = np.mean(flow[..., 0])
            avg_flow_y = np.mean(flow[..., 1])
            global_motion = np.array([avg_flow_x, avg_flow_y])
            relative_flow = flow - global_motion

            relative_mag, _ = cv2.cartToPolar(relative_flow[..., 0], relative_flow[..., 1])
            motion_mask = cv2.threshold(relative_mag, motion_threshold, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            filtered_contours = []
            for contour in contours:
                if cv2.contourArea(contour) < min_contour_area:
                    continue

                mask = np.zeros_like(relative_mag, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)

                contour_flow_x = cv2.mean(relative_flow[..., 0], mask=mask)[0]
                contour_flow_y = cv2.mean(relative_flow[..., 1], mask=mask)[0]
                average_flow = np.sqrt(contour_flow_x ** 2 + contour_flow_y ** 2)

                if average_flow >= 1.0:
                    filtered_contours.append(contour)

            filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:3]
            detections = []

            # 仅提取中心点
            for contour in filtered_contours:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append((x + w // 2, y + h // 2))  # 中心点用于跟踪

            # 使用中心点更新跟踪器
            updated_positions = tracker.update_trackers(detections)

            # 将每个跟踪器的位置和对应的 ID 写入文件
            for tracker_id, position in updated_positions.items():
                for contour in filtered_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if position == (x + w // 2, y + h // 2):  # 匹配位置，确保 ID 唯一
                        f.write(f"{frame_number},{x},{y},{w},{h},{tracker_id}\n")
                        save_cropped_object(frame, position, (w, h), tracker_id, frame_number, output_dir)
                        break  # 跳出循环，防止重复写入

            # 控制台输出进度，每 100 帧输出一次
            if frame_number % 100 == 0:
                remaining_frames = total_frames - frame_number
                print(f"Processed frame {frame_number}/{total_frames}, {remaining_frames} frames remaining.")

            prev_gray = curr_gray

    input_source.release()
    print(f"Processing completed. Results saved to '{output_file}'.")


def process_folder(i, dirnames):
    # 子目录路径
    child_path = os.path.join(dirnames, "img")
    child_dir = os.path.join(root_path, child_path)

    # 初始化输入源并运行跟踪
    input_source = ImageSequenceInput(child_dir)
    run_tracking(
        input_source,
        output_file=f"tracking_results_{i}.txt",
        output_dir=f"output_large/video_{i}"
    )

if __name__ == "__main__":
    # input_source = VideoInput("video.mp4")
    counter = 0
    lock = Lock()  # 用于线程同步
    root_path="../Codes/Data/TinyTLP_V2"
    dirnames_list = os.listdir(root_path)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for dirnames in dirnames_list:
            # 使用锁保护计数器
            with lock:
                counter += 1
                current_index = counter
                if counter==2:
                    break

            # 提交任务到线程池
            futures.append(executor.submit(process_folder, current_index, dirnames))

        # 等待所有线程完成
        for future in futures:
            future.result()  # 如果需要捕获异常，可以在这里处理

    # input_source = ImageSequenceInput("img")
    # run_tracking(input_source)
