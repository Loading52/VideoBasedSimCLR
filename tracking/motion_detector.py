import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os
import time
from utils import VideoInput, ImageSequenceInput
import hdbscan


class MotionDetector:
    def __init__(self, eps=5, min_samples=50, magnitude_threshold=2.0,
                 pixel_threshold=(10, 1000), angle_variance_threshold=0.1):
        """
        初始化运动检测器。
        :param eps: DBSCAN 聚类参数，邻域半径。
        :param min_samples: DBSCAN 聚类参数，最小样本数。
        :param magnitude_threshold: 运动强度阈值，用于过滤光流中的弱运动。
        :param pixel_threshold: 连通域的像素数量阈值 (min, max)。
        :param angle_variance_threshold: 方向一致性的方差阈值。
        """
        self.eps = eps
        self.min_samples = min_samples
        self.magnitude_threshold = magnitude_threshold
        self.pixel_threshold = pixel_threshold
        self.angle_variance_threshold = angle_variance_threshold

    def detect_motion(self, prev_frame, curr_frame):
        """
        检测一帧图像中的运动目标。
        :param prev_frame: 上一帧灰度图像。
        :param curr_frame: 当前帧灰度图像。
        :return: 检测结果，包含中心点、大小、平均位移的字典数组。
        """
        # 计算光流
        dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        flow = dis_flow.calc(prev_frame, curr_frame, None)

        # 去除全局运动
        avg_flow_x = np.mean(flow[..., 0])
        avg_flow_y = np.mean(flow[..., 1])
        global_motion = np.array([avg_flow_x, avg_flow_y])
        flow -= global_motion

        # 提取光流的幅度和方向
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 根据运动强度阈值过滤
        motion_mask = magnitude > self.magnitude_threshold
        motion_points = np.column_stack(np.where(motion_mask))

        if motion_points.size == 0:
            print("No motion detected.")
            return []  # 没有检测到运动

        if motion_points.size / motion_mask.size > 0.2:
            print("Too much motion detected.")
            return []

        self.visualize_binary_flow(motion_mask)

        # 标记运动区域
        labels = self.label_motion_regions(motion_points, angle, magnitude, motion_mask)

        # 构建检测结果
        return self.generate_detection_results(motion_points, labels, flow)

    def cluster_motion(self, motion_points, angle, magnitude, motion_mask):
        # 构造聚类特征：位置 + 光流大小 + 光流方向
        motion_features = np.hstack((
            motion_points,  # 位置 [x, y]
            magnitude[motion_mask][:, None],  # 光流大小
            angle[motion_mask][:, None]  # 光流方向
        ))
        # 使用 DBSCAN 聚类提取运动目标
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(motion_features)
        labels = clustering.labels_
        # clustering = hdbscan.HDBSCAN(min_samples=50, min_cluster_size=100, metric='euclidean')
        # labels = clustering.fit_predict(motion_features)
        return labels

    def label_motion_regions(self, motion_points, angle, magnitude, motion_mask):
        """
        使用连通性检测和方向一致性判断标记运动区域。
        :param motion_points: 运动点的位置坐标 [x, y]。
        :param angle: 光流方向的角度矩阵。
        :param magnitude: 光流的大小矩阵。
        :param motion_mask: 二值掩码，表示运动点的位置。
        :return: 每个运动点的标签数组（与 motion_points 对应）。
        """
        # 连通性分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask.astype(np.uint8),
                                                                                connectivity=8)

        # 初始化返回的标签
        point_labels = -np.ones(len(motion_points), dtype=int)  # 默认值为 -1 表示未分配

        # 遍历每个连通区域
        for label in range(1, num_labels):  # 忽略背景区域（标签 0）
            region_mask = (labels == label)

            # 获取连通域的像素数量
            region_area = stats[label, cv2.CC_STAT_AREA]

            # 如果区域像素数小于阈值，跳过
            if region_area < self.pixel_threshold[0] or region_area > self.pixel_threshold[1]:
                continue

            # 获取该连通域中的运动点索引
            region_indices = np.where(labels[motion_points[:, 0], motion_points[:, 1]] == label)[0]

            # 计算方向的一致性
            region_angles = angle[region_mask]
            angle_variance = np.var(region_angles)

            # 如果方向一致性较高（方差小于阈值），认为是有效目标
            if angle_variance < self.angle_variance_threshold:
                point_labels[region_indices] = label

        return point_labels

    def generate_detection_results(self, motion_points, labels, flow):
        """
        根据标记生成检测结果。
        :param motion_points: 光流检测到的运动点。
        :param labels: 运动点对应的区域标签。
        :param flow: 光流场。
        :return: 检测结果的字典数组。
        """
        unique_labels = set(labels)
        detection_results = []

        for label in unique_labels:
            if label == -1:  # 忽略噪声点
                continue
            cluster_points = motion_points[labels == label]

            if cluster_points.size == 0:
                continue

            # 计算中心点
            center_y, center_x = np.mean(cluster_points, axis=0)

            # 计算边界框大小
            x_min, y_min = cluster_points.min(axis=0)
            x_max, y_max = cluster_points.max(axis=0)
            size = (x_max - x_min, y_max - y_min)

            # 计算平均位移
            displacements = flow[cluster_points[:, 0], cluster_points[:, 1]]
            avg_displacement = np.mean(displacements, axis=0)

            # 保存检测结果
            detection_results.append({
                "center": (center_x, center_y),
                "size": size,
                "avg_displacement": avg_displacement
            })

        return detection_results

    @staticmethod
    def visualize_binary_flow(motion_mask):
        """
        显示二值化后的光流场。
        :param motion_mask: 二值化后的光流场掩码。
        """
        binary_flow_image = (motion_mask * 255).astype(np.uint8)
        cv2.imshow(f"Binary Flow", binary_flow_image)

    @staticmethod
    def visualize_motion(frame, motion_points, labels, output_folder, frame_count):
        """
        可视化运动目标并保存结果（可选）。
        :param frame: 当前帧图像。
        :param motion_points: 光流检测到的运动点位置。
        :param labels: DBSCAN 聚类的标签。
        :param output_folder: 结果保存的文件夹（如果指定）。
        :param frame_count: 当前帧的编号。
        :return: 带运动目标标记的帧。
        """
        frame_with_motion = frame.copy()
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # 忽略噪声点
                continue
            cluster_points = motion_points[labels == label]
            x_min, y_min = cluster_points.min(axis=0)
            x_max, y_max = cluster_points.max(axis=0)
            cv2.rectangle(frame_with_motion, (y_min, x_min), (y_max, x_max), (0, 255, 0), 2)
        cv2.imshow("Motion Detection", frame_with_motion)

        # 保存检测结果
        if output_folder:
            cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count:04d}.jpg"), frame_with_motion)

        return frame_with_motion


def main():
    # video_input = VideoInput("video.mp4")
    video_input = ImageSequenceInput("img1")
    motion_detector = MotionDetector(eps=10, min_samples=100, magnitude_threshold=2, pixel_threshold=(400, 200000), angle_variance_threshold=10)

    prev_frame = None
    while True:
        frame = video_input.read_frame()
        if frame is None:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            results = motion_detector.detect_motion(prev_frame, frame_gray)

            # 在原图上绘制检测结果
            for result in results:
                center = tuple(map(int, result["center"]))
                size = result["size"]
                x_min, y_min = int(center[0] - size[0] / 2), int(center[1] - size[1] / 2)
                x_max, y_max = int(center[0] + size[0] / 2), int(center[1] + size[1] / 2)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Displ: {result['avg_displacement']}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Motion Detection", frame)

        prev_frame = frame_gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 保存最后一帧
    cv2.imwrite("motion_detection_result.jpg", frame)

    video_input.release()
    cv2.destroyAllWindows()


# 测试代码
if __name__ == "__main__":
    main()
