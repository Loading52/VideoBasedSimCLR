from motion_detector import MotionDetector
from object_tracking import ObjectTracker
from tracking.resnet import ResNet50  # 替换为你的 ResNet50 实现
import torch
import cv2
from utils import VideoInput, ImageSequenceInput

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50(cifar_head=False)
state_dict = torch.load("resnet50_imagenet_bs2k_epochs200.pth")
model.load_state_dict(state_dict)
model = model.to(device)

# 初始化 MotionDetector 和 ObjectTracker
motion_detector = MotionDetector(eps=10, min_samples=100, magnitude_threshold=2, pixel_threshold=(400, 200000), angle_variance_threshold=10)
object_tracker = ObjectTracker(model, device=device, confidence_threshold=0.5, crop_ratio=1)

# 视频读取
# video_input = VideoInput("video.mp4")
video_input = ImageSequenceInput("img1")

# 获取第一帧
prev_frame = video_input.read_frame()
if prev_frame is None:
    raise ValueError("Video contains no frames.")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# 获取第二帧
current_frame = video_input.read_frame()
if current_frame is None:
    raise ValueError("Video contains only one frame.")

current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

# 初始检测运动区域
motion_results = motion_detector.detect_motion(prev_gray, current_gray)

# 初始化每个物体的边界框
objects = []
for result in motion_results:
    bbox = (
        int(result["center"][0] - result["size"][0] / 2),
        int(result["center"][1] - result["size"][1] / 2),
        int(result["size"][0]),
        int(result["size"][1])
    )
    objects.append({"bbox": bbox, "lost": False})

# 逐帧跟踪
while True:
    current_frame = video_input.read_frame()
    if current_frame is None:
        break

    # 跟踪每个物体
    for obj in objects:
        if obj["lost"]:
            continue

        bbox = obj["bbox"]
        new_bbox, confidence = object_tracker.track(prev_frame, current_frame, bbox)

        if confidence > object_tracker.confidence_threshold:
            x, y, w, h = new_bbox
            obj["bbox"] = new_bbox
            cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(current_frame, f"Conf: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            obj["lost"] = True
            print("Tracking lost for object.")

    cv2.imshow("Tracking", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_frame = current_frame

# 保存最后一帧
cv2.imwrite("object_tracking_result0.jpg", current_frame)

video_input.release()
cv2.destroyAllWindows()
