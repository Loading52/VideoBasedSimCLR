# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Data pipeline."""

import functools
from absl import flags
from absl import logging

import data_util
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS


def build_input_fn_custom(data_dir, global_batch_size, topology, is_training, num_classes):
    """构建自定义输入函数，保留伪标签."""

    def _input_fn(input_context):
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        logging.info('Global batch size: %d', global_batch_size)
        logging.info('Per-replica batch size: %d', batch_size)

        preprocess_fn = get_preprocess_fn(is_training, is_pretrain=True)

        def map_fn(file_path):
            """
            从文件夹生成正样本对，并生成one-hot编码的标签（videoId-objectId）。
            Args:
                file_path: 每个 object_xxx 文件夹路径。
            Returns:
                img_pair: 两张增强后的图片拼接作为正样本对。
                label: 唯一标识符的 one-hot 编码。
            """
            # 解析 videoId 和 objectId
            # 替换路径中的反斜杠为正斜杠
            # folder_path = tf.strings.regex_replace(folder_path, r"\\", "/")

            # 打印替换后的路径
            parts = tf.strings.split(file_path, '\\')[-3:]  # video_xxx/object_xxx
            video_id = tf.strings.to_number(tf.strings.substr(parts[0], 6, -1), out_type=tf.int32)
            object_id = tf.strings.to_number(tf.strings.substr(parts[1], 7, -1), out_type=tf.int32)
            image_label = video_id * 1000 + object_id  # 确保唯一性

            # 转换为 one-hot 编码
            image_label = tf.one_hot(image_label, depth=num_classes)


            folder_path = tf.strings.regex_replace(file_path, r"\\frame_.*\.jpg$", "")
            all_files = tf.io.matching_files(folder_path + "\\*.jpg")
            shuffled_files = tf.random.shuffle(all_files)
            filtered_files = tf.boolean_mask(shuffled_files, shuffled_files != file_path)

            img1 = preprocess_file(file_path, preprocess_fn)
            # 随机选取另一张图片作为正样本对
            img2_path = tf.cond(tf.size(filtered_files) > 0,
                                lambda: filtered_files[tf.random.uniform(shape=(), maxval=tf.size(filtered_files), dtype=tf.int32)],
                                lambda: file_path)  # 如果没有其他图片，则选自己
            img2 = preprocess_file(img2_path, preprocess_fn)

            # 从文件夹中加载图像文件
            # image_files = tf.io.matching_files(image_path + "\\*.jpg")
            # shuffled_files = tf.random.shuffle(image_files)
            # file_count = tf.shape(shuffled_files)[0]
            # if file_count >= 2:
            #     # 文件夹中有足够的图片，直接加载两张不同的图片
            #     img1 = preprocess_file(shuffled_files[0], preprocess_fn)
            #     img2 = preprocess_file(shuffled_files[1], preprocess_fn)
            # else:
            #     # 文件夹中图片不足，使用增强方法生成两种视图
            #     # tf.print(folder_path+" has less than 2 images. Using augmentation.")
            #     img = preprocess_file(shuffled_files[0], preprocess_fn)
            #     img1 = preprocess_fn(img)  # 第一次增强
            #     img2 = preprocess_fn(img)  # 第二次增强（不同随机增强参数）
            #
            image_pair = tf.concat([img1, img2], axis=-1)  # 拼接正样本对

            return image_pair, image_label

        def preprocess_file(file_path, preprocess_f):
            """加载并预处理单张图像."""
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = preprocess_f(img)
            return img

        # 遍历所有 object_xxx 文件夹
        dataset = tf.data.Dataset.list_files(f"{data_dir}/video_*/object_*/frame_*.jpg", shuffle=is_training)
        dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataset = dataset.map(map_fn)

        if is_training:
            buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
            dataset = dataset.shuffle(buffer_size=batch_size*buffer_multiplier).repeat(-1)

        dataset = dataset.batch(batch_size, drop_remainder=is_training)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # 检查数据集的输出
        for img_pair, label in dataset.take(1):  # 打印第一个批次数据
            tf.print("Batch Image Pair Shape:", tf.shape(img_pair))
            tf.print("Batch Label Shape:", tf.shape(label))
            tf.print("Batch Label Example:", label[0])

        return dataset

    return _input_fn


def build_input_fn(builder, global_batch_size, topology, is_training):
  """Build input function.

  Args:
    builder: TFDS builder for specified dataset.
    global_batch_size: Global batch size.
    topology: An instance of `tf.tpu.experimental.Topology` or None.
    is_training: Whether to build in training mode.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """

  def _input_fn(input_context):
    """Inner input function."""
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    logging.info('Global batch size: %d', global_batch_size)
    logging.info('Per-replica batch size: %d', batch_size)
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False)
    num_classes = builder.info.features['label'].num_classes

    def map_fn(image, label):
      """Produces multiple transformations of the same batch."""
      if is_training and FLAGS.train_mode == 'pretrain':
        xs = []
        for _ in range(2):  # Two transformations
          xs.append(preprocess_fn_pretrain(image))
        image = tf.concat(xs, -1)
      else:
        image = preprocess_fn_finetune(image)
      label = tf.one_hot(label, num_classes)
      return image, label

    logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
    dataset = builder.as_dataset(
        split=FLAGS.train_split if is_training else FLAGS.eval_split,
        shuffle_files=is_training,
        as_supervised=True,
        # Passing the input_context to TFDS makes TFDS read different parts
        # of the dataset on different workers. We also adjust the interleave
        # parameters to achieve better performance.
        read_config=tfds.ReadConfig(
            interleave_cycle_length=32,
            interleave_block_length=1,
            input_context=input_context))
    if FLAGS.cache_dataset:
      dataset = dataset.cache()
    if is_training:
      options = tf.data.Options()
      options.experimental_deterministic = False
      options.experimental_slack = True
      dataset = dataset.with_options(options)
      buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
      dataset = dataset.shuffle(batch_size * buffer_multiplier)
      dataset = dataset.repeat(-1)
    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  return _input_fn


def build_distributed_dataset(builder, batch_size, is_training, strategy,
                              topology, num_classes=None):
  if num_classes is not None:
    input_fn = build_input_fn_custom(builder, batch_size, topology, is_training, num_classes)
  else:
    input_fn = build_input_fn(builder, batch_size, topology, is_training)
  return strategy.distribute_datasets_from_function(input_fn)


def get_preprocess_fn(is_training, is_pretrain):
  """Get function that accepts an image and returns a preprocessed image."""
  # Disable test cropping for small images (e.g. CIFAR)
  if FLAGS.image_size <= 32:
    test_crop = False
  else:
    test_crop = True
  color_jitter_strength = FLAGS.color_jitter_strength if is_pretrain else 0.
  return functools.partial(
      data_util.preprocess_image,
      height=FLAGS.image_size,
      width=FLAGS.image_size,
      is_training=is_training,
      color_jitter_strength=color_jitter_strength,
      test_crop=test_crop)
