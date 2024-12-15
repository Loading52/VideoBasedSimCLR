import tensorflow as tf

# 加载检查点
checkpoint = tf.train.load_checkpoint("./pretrained/r50_1x_sk0")
variable_names = checkpoint.get_variable_to_shape_map()

# 分类变量
conv_vars = {name: shape for name, shape in variable_names.items() if "conv" in name}
dense_vars = {name: shape for name, shape in variable_names.items() if "dense" in name}
batch_norm_vars = {name: shape for name, shape in variable_names.items() if "batch_normalization" in name}

print("Convolutional Layers:")
for name, shape in conv_vars.items():
    print(f"{name}: {shape}")

print("\nDense Layers:")
for name, shape in dense_vars.items():
    print(f"{name}: {shape}")

print("\nBatch Normalization Layers:")
for name, shape in batch_norm_vars.items():
    print(f"{name}: {shape}")
