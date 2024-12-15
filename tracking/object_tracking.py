from tracking.resnet import ResNet50  # 模型在 tracking 文件夹中
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision.transforms import transforms
import time


import torch
import torch.nn.functional as F
import cv2
import time
from torchvision.transforms import transforms


class ObjectTracker:
    def __init__(self, model, device="cuda", confidence_threshold=0.5, crop_ratio=0.7):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.confidence_threshold = confidence_threshold
        self.crop_ratio = crop_ratio

    def extract_features(self, frame, bbox):
        x, y, w, h = bbox
        # 按比例缩小边界框
        new_w = int(w * self.crop_ratio)
        new_h = int(h * self.crop_ratio)
        new_x = x + (w - new_w) // 2
        new_y = y + (h - new_h) // 2
        x, y, w, h = new_x, new_y, new_w, new_h

        crop = frame[y:y+h, x:x+w]
        crop_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(crop_tensor)
        return features

    def track(self, prev_frame, current_frame, prev_bbox):
        prev_features = self.extract_features(prev_frame, prev_bbox)
        x, y, w, h = prev_bbox
        search_radius = 20
        step = 5
        candidates = []

        for dx in range(-search_radius, search_radius + 1, step):
            for dy in range(-search_radius, search_radius + 1, step):
                candidate_bbox = (x + dx, y + dy, w, h)
                if candidate_bbox[0] < 0 or candidate_bbox[1] < 0:
                    continue
                candidates.append(candidate_bbox)

        candidate_tensors = []
        for candidate_bbox in candidates:
            cx, cy, cw, ch = candidate_bbox

            # 检查候选框是否有效
            if cw <= 10 or ch <= 10:
                continue  # 宽或高无效则跳过

            # 检查裁剪范围是否在图像边界内
            if cy < 10 or cx < 10 or (cy + ch) > current_frame.shape[0] - 10 or (cx + cw) > current_frame.shape[1] - 10:
                continue  # 跳过越界候选框

            crop = current_frame[cy:cy + ch, cx:cx + cw]
            crop_tensor = self.transform(crop).unsqueeze(0).to(self.device)
            candidate_tensors.append(crop_tensor)

        if candidate_tensors:
            batch_tensor = torch.cat(candidate_tensors, dim=0)
            with torch.no_grad():
                candidate_features = self.model(batch_tensor)

            similarities = F.cosine_similarity(prev_features, candidate_features)
            best_index = torch.argmax(similarities).item()
            best_bbox = candidates[best_index]
            confidence = similarities[best_index].item()
        else:
            best_bbox = prev_bbox
            confidence = 0.0

        return best_bbox, confidence


def main():
    # ** 模型加载代码 **
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = ResNet50(cifar_head=False)

    # 加载权重
    state_dict = torch.load("resnet50_imagenet_bs2k_epochs200.pth")
    model.load_state_dict(state_dict)

    # 将模型加载到 GPU
    model = model.to(device)
    model.eval()

    # 初始化跟踪器
    tracker = ObjectTracker(model, confidence_threshold=0.1, crop_ratio=0.7)

    # 视频测试代码
    video_path = "video.mp4"  # 输入视频路径
    cap = cv2.VideoCapture(video_path)

    # 假设物体在第一帧的位置和大小已知
    _, first_frame = cap.read()
    initial_bbox = (100, 100, 50, 50)  # 示例边界框 (x, y, w, h)

    prev_frame = first_frame
    prev_bbox = initial_bbox

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        # 跟踪物体
        current_bbox, confidence = tracker.track(prev_frame, current_frame, prev_bbox)

        # 检查可信度
        if confidence < tracker.confidence_threshold:
            print(f"Tracking lost at frame {tracker.total_frames}, confidence: {confidence:.4f}")
            break

        # 可视化跟踪结果
        x, y, w, h = current_bbox
        cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(current_frame, f"Confidence: {confidence:.2f}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Tracking", current_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 更新上一帧
        prev_frame = current_frame
        prev_bbox = current_bbox

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()