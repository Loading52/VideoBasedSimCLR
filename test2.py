import tensorflow as tf
import numpy as np
import cv2
import time
import tensorflow_io as tfio


class SimCLRTester:
    def __init__(self, saved_model_dir):
        """
        初始化 SimCLR 模型测试器。
        :param saved_model_dir: SavedModel 文件夹路径。
        """
        self.model = self._load_saved_model(saved_model_dir)

    def _load_saved_model(self, saved_model_dir):
        """
        加载 SavedModel。
        :param saved_model_dir: SavedModel 文件夹路径。
        :return: 已加载的模型。
        """
        print(f"Loading SavedModel from: {saved_model_dir}")
        model = tf.saved_model.load(saved_model_dir)
        print("Model loaded successfully!")
        print("Available signatures:", list(model.signatures.keys()))
        return model

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
        # 使用默认签名进行推理
        signature = self.model.signatures["serving_default"]
        outputs = signature(image_tensor)
        return outputs

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
        signature = self.model.signatures["serving_default"]
        total_time = 0
        for _ in range(num_iterations):
            start_time = time.time()
            signature(image_tensor)
            total_time += time.time() - start_time

        print(f"Average inference time: {total_time / num_iterations:.4f} seconds")


if __name__ == "__main__":
    # SavedModel 路径
    SAVED_MODEL_DIR = "gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/r50_1x_sk0/saved_model/"  # 替换为您的 SavedModel 文件夹路径

    # 测试图像路径
    TEST_IMAGE_PATH = "output/frame_0001.jpg"  # 替换为您的测试图像路径

    # 初始化测试器
    tester = SimCLRTester(SAVED_MODEL_DIR)

    # 提取特征
    print("Extracting features...")
    features = tester.extract_features(TEST_IMAGE_PATH)
    print("Feature outputs:", features)

    # 测试推理时间
    print("Testing inference time...")
    tester.test_inference_time(TEST_IMAGE_PATH, num_iterations=10)
