import cv2
import numpy as np


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

        # 初始化状态向量，包括位置和速度（初始速度设为0）
        kf.statePre = np.array([initial_position[0], initial_position[1], 0, 0], dtype=np.float32)
        kf.statePost = np.array([initial_position[0], initial_position[1], 0, 0], dtype=np.float32)

        # 使用初始位置更新卡尔曼滤波器的测量值
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

            # 如果没有匹配的检测，使用预测的位置
            if updated_position is None:
                updated_position = predicted_position

            # 更新位置和历史轨迹
            tracker_info["position"] = updated_position
            tracker_info["history"].append(updated_position)

            # 限制轨迹历史的长度
            if len(tracker_info["history"]) > self.max_history:
                tracker_info["history"].pop(0)

            # 调试信息，确保 `updated_position` 和 `predicted_position` 正确
            print(
                f"Tracker ID: {tracker_id}, Predicted Position: {predicted_position}, Updated Position: {updated_position}")
            print(f"History: {tracker_info['history']}")

            updated_positions[tracker_id] = updated_position

        # 添加新检测的物体
        for detection in detections:
            tracker_id = self.add_tracker(detection)
            updated_positions[tracker_id] = detection

        return updated_positions

    def match(self, predicted, detection, threshold=50):
        distance = np.linalg.norm(np.array(predicted) - np.array(detection))
        return distance < threshold

    def get_tracks(self):
        return {tracker_id: tracker_info["history"] for tracker_id, tracker_info in self.trackers.items()}


# 加载视频和处理视频帧
cap = cv2.VideoCapture('video.mp4')
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
motion_threshold = 2.0
min_contour_area = 500
tracker = ObjectTracker()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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
    detections = []

    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2
        detections.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    updated_positions = tracker.update_trackers(detections)

    for tracker_id, position in updated_positions.items():
        cv2.circle(frame, position, 5, (0, 255, 0), -1)
        history = tracker.trackers[tracker_id]["history"]

        # print ("len:", len(history))
        # for i in range(0, len(history)):
        #     print(history[i])

        for i in range(1, len(history)):
            pt1 = history[i - 1]
            pt2 = history[i]
            # print(pt1, pt2)
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

    cv2.imshow("Object Tracking with Global Motion Compensation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = curr_gray

cap.release()
cv2.destroyAllWindows()
