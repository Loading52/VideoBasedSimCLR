import cv2
import random
from ObjectTracker import VideoInput, ImageSequenceInput  # 从原文件导入


def get_color_for_id(obj_id, id_colors):
    """获取物体ID对应的颜色，如果没有则生成随机颜色"""
    if obj_id not in id_colors:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        id_colors[obj_id] = color
    return id_colors[obj_id]


def visualize_tracking(input_source, tracking_data_file, output_video="output_visualization.mp4"):
    # 获取输入源的第一帧，确定视频属性
    first_frame = input_source.read_frame()
    if first_frame is None:
        print("Failed to read from input source.")
        return

    height, width = first_frame.shape[:2]
    fps = 30  # 视频写入帧率，实际播放控制通过cv2.waitKey

    # 初始化视频写入器
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 读取跟踪数据文件
    tracking_data = {}
    with open(tracking_data_file, "r") as f:
        for line in f:
            frame_number, x, y, w, h, obj_id = map(int, line.strip().split(','))
            if frame_number not in tracking_data:
                tracking_data[frame_number] = []
            tracking_data[frame_number].append((x, y, w, h, obj_id))

    # 创建一个字典来存储每个物体ID的颜色
    id_colors = {}

    # 可视化结果
    frame_number = 0
    while True:
        frame = input_source.read_frame()
        if frame is None:
            break

        frame_number += 1

        # 检查当前帧是否有跟踪数据
        if frame_number in tracking_data:
            for (x, y, w, h, obj_id) in tracking_data[frame_number]:
                # 获取或分配给该ID的颜色
                color = get_color_for_id(obj_id, id_colors)

                # 画出矩形框和物体ID
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID: {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 将帧写入输出视频
        out.write(frame)
        cv2.imshow("Tracking Visualization", frame)

        # 增加延迟时间，控制播放速度
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # 释放资源
    input_source.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 可使用 VideoInput 或 ImageSequenceInput 作为输入
    # video_source = VideoInput("video.mp4")
    image_source = ImageSequenceInput("img")  # 指定图片序列文件夹
    visualize_tracking(image_source, "tracking_results.txt")
