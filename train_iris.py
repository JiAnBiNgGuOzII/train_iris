import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 加载鸢尾花数据集
print("加载数据集...")
iris = load_iris()
X = iris.data   # 样本特征：150行 x 4列（花萼长、宽，花瓣长、宽）
y = iris.target # 样本标签：150行 x 1列（0: Setosa, 1: Versicolour, 2: Virginica）

# 2. 划分训练集和测试集
# test_size=0.2 表示 20% 的数据（30组）作为测试集，80%（120组）作为训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 构建全连接神经网络模型
print("构建神经网络...")
model = Sequential([
    # 输入层与第一个隐藏层：输入维度为4，隐藏层包含16个神经元，激活函数为ReLU
    Dense(16, activation='relu', input_shape=(4,)),
    # 第二个隐藏层：包含8个神经元，激活函数为ReLU
    Dense(8, activation='relu'),
    # 输出层：包含3个神经元（对应3个类别），激活函数为Softmax
    Dense(3, activation='softmax')
])

# 4. 编译模型
# 优化器使用 adam，损失函数使用稀疏分类交叉熵
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 训练模型
print("开始训练模型...")
# epochs=100 表示整个训练集将被完整遍历100次
# batch_size=8 表示每次计算梯度并更新权重时使用8个样本
model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1)

# 6. 在测试集上验证模型准确率
print("评估模型...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"测试集准确率: {accuracy * 100:.2f}%")

# # 7. 导出模型为 .h5 格式，供 STM32Cube.AI 使用
# model_filename = 'iris_model.h5'
# model.save(model_filename)
# print(f"模型已成功保存为 {model_filename}")

# 7. 导出模型为 TensorFlow Lite (.tflite) 格式
print("正在转换为 TFLite 格式...")
# 使用 TFLiteConverter 从现有的 Keras 模型进行转换
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 将转换后的字节流写入 .tflite 文件
tflite_filename = 'iris_model.tflite'
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model)

print(f"模型已成功保存为 {tflite_filename}")