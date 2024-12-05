import tensorflow as tf
import numpy as np
import cv2
import time


class SimCLRCheckpointTester:
    def __init__(self, checkpoint_dir, checkpoint_prefix):
        """
        初始化 SimCLR 模型检查点测试器。
        :param checkpoint_dir: 检查点文件夹路径。
        :param checkpoint_prefix: 检查点文件前缀（例如 model.ckpt-250228）。
        """
        self.model = self._build_model()  # 构建模型结构
        self._restore_checkpoint(checkpoint_dir, checkpoint_prefix)  # 恢复检查点

    def _build_model(self):
        """
        构建与检查点匹配的模型结构。
        :return: 构建的 Keras 模型。
        """
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,  # 无预训练权重
            pooling='avg',  # 使用全局平均池化层
            input_shape=(224, 224, 3)
        )
        return base_model

    def _restore_checkpoint(self, checkpoint_dir, checkpoint_prefix):
        """
        从检查点恢复权重到模型中。
        :param checkpoint_dir: 检查点文件夹路径。
        :param checkpoint_prefix: 检查点文件前缀。
        """
        checkpoint_path = f"{checkpoint_dir}/{checkpoint_prefix}"
        checkpoint = tf.train.Checkpoint(model=self.model)

        # 恢复权重
        checkpoint.restore(checkpoint_path).expect_partial()
        print(f"Model restored from checkpoint: {checkpoint_path}")

    def extract_features(self, image_path):
        """
        提取图像的特征。
        :param image_path: 输入图像路径。
        :return: 特征向量。
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        # 图像预处理
        image_resized = cv2.resize(image, (224, 224)) / 255.0  # 调整到 224x224
        image_tensor = tf.convert_to_tensor(np.expand_dims(image_resized, axis=0), dtype=tf.float32)

        # 推理，获取特征向量
        features = self.model(image_tensor, training=False)
        return features.numpy()

    def test_inference_time(self, image_path, num_iterations=10):
        """
        测试模型的推理时间。
        :param image_path: 测试图像路径。
        :param num_iterations: 测试次数。
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        # 图像预处理
        image_resized = cv2.resize(image, (224, 224)) / 255.0
        image_tensor = tf.convert_to_tensor(np.expand_dims(image_resized, axis=0), dtype=tf.float32)

        # 测试推理时间
        total_time = 0
        for _ in range(num_iterations):
            start_time = time.time()
            self.model(image_tensor, training=False)
            total_time += time.time() - start_time

        print(f"Average inference time: {total_time / num_iterations:.4f} seconds")


if __name__ == "__main__":
    # 检查点路径
    CHECKPOINT_DIR = "./pretrained/r50_1x_sk0"
    CHECKPOINT_PREFIX = "model.ckpt-250228"

    # 测试图像路径
    TEST_IMAGE_PATH = "output/frame_0001.jpg"

    # 初始化测试器
    tester = SimCLRCheckpointTester(CHECKPOINT_DIR, CHECKPOINT_PREFIX)

    # 提取特征
    print("Extracting features...")
    features = tester.extract_features(TEST_IMAGE_PATH)
    print("Feature vector shape:", features.shape)

    # 测试推理时间
    print("Testing inference time...")
    tester.test_inference_time(TEST_IMAGE_PATH, num_iterations=10)
