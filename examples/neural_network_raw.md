# 神经网络示例
使用TensorFlow v2构建一个两层隐藏层完全连接的神经网络（多层感知器）。

这个例子使用低级方法来更好地理解构建神经网络和训练过程背后的所有机制。

- 作者: Aymeric Damien
- 原项目: https://github.com/aymericdamien/TensorFlow-Examples/

### 神经网络概述
![](http://qiniu.aihubs.net/blog/20190918/VrYoKjla7ug2.jpeg?imageslim)

### MNIST 数据集概述
此示例使用手写数字的MNIST数据集。该数据集包含60,000个用于训练的示例和10,000个用于测试的示例。这些数字已经过尺寸标准化并位于图像中心，图像是固定大小(28x28像素)，值为0到255。

在此示例中，每个图像将转换为float32并归一化为[0,1]，并展平为784个特征的一维数组（28 * 28）

![](http://qiniu.aihubs.net/blog/20190918/t9VTbvdh68hy.png?imageslim)

更多信息请查看链接: http://yann.lecun.com/exdb/mnist/

```python
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
```

```python
# MNIST 数据集参数
num_classes = 10 # 所有类别（数字 0-9）
num_features = 784 # 数据特征数目 (图像形状: 28*28)

# 训练参数
learning_rate = 0.001
training_steps = 3000
batch_size = 256
display_step = 100

# 网络参数
n_hidden_1 = 128 # 第一层隐含层神经元的数目
n_hidden_2 = 256 # 第二层隐含层神经元的数目
```

```python
# 准备MNIST数据
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 转化为float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# 将每张图像展平为具有784个特征的一维向量（28 * 28）
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# 将图像值从[0,255]归一化到[0,1]
x_train, x_test = x_train / 255., x_test / 255.
```

```python
# 使用tf.data API对数据进行随机排序和批处理
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
```

```python
# 存储层的权重和偏置

# 随机值生成器初始化权重
random_normal = tf.initializers.RandomNormal()

weights = {
    'h1': tf.Variable(random_normal([num_features, n_hidden_1])),
    'h2': tf.Variable(random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([num_classes]))
}
```

```python
# 创建模型
def neural_net(x):
    # Hidden fully connected layer with 128 neurons.
    # 具有128个神经元的隐含完全连接层
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Apply sigmoid to layer_1 output for non-linearity.
    # 将sigmoid用于layer_1输出以获得非线性
    layer_1 = tf.nn.sigmoid(layer_1)
    
    # 具有128个神经元的隐含完全连接层
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # 将sigmoid用于layer_2输出以获得非线性
    layer_2 = tf.nn.sigmoid(layer_2)
    
    # 输出完全连接层，每一个神经元代表一个类别
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    # 应用softmax将输出标准化为概率分布
    return tf.nn.softmax(out_layer)
```

```python
# 交叉熵损失函数
def cross_entropy(y_pred, y_true):
    # 将标签编码为独热向量
    y_true = tf.one_hot(y_true, depth=num_classes)
    # 将预测值限制在一个范围之内以避免log（0）错误
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # 计算交叉熵
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

#  准确率评估
def accuracy(y_pred, y_true):
     # 预测类是预测向量中最高分的索引（即argmax）
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# 随机梯度下降优化器
optimizer = tf.optimizers.SGD(learning_rate)
```

```python
# 优化过程
def run_optimization(x, y):
    # 将计算封装在GradientTape中以实现自动微分
    with tf.GradientTape() as g:
        pred = neural_net(x)
        loss = cross_entropy(pred, y)
        
    # 要更新的变量，即可训练的变量
    trainable_variables = weights.values() + biases.values()

    # 计算梯度
    gradients = g.gradient(loss, trainable_variables)
    
    # 按gradients更新 W 和 b
    optimizer.apply_gradients(zip(gradients, trainable_variables))
```

```python
# 针对给定步骤数进行训练
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # 运行优化以更新W和b值
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0:
        pred = neural_net(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
```
**output:**
```
step: 100, loss: 567.292969, accuracy: 0.136719
step: 200, loss: 398.614929, accuracy: 0.562500
step: 300, loss: 226.743774, accuracy: 0.753906
step: 400, loss: 193.384521, accuracy: 0.777344
step: 500, loss: 138.649963, accuracy: 0.886719
step: 600, loss: 109.713669, accuracy: 0.898438
step: 700, loss: 90.397217, accuracy: 0.906250
step: 800, loss: 104.545380, accuracy: 0.894531
step: 900, loss: 94.204697, accuracy: 0.890625
step: 1000, loss: 81.660645, accuracy: 0.906250
step: 1100, loss: 81.237137, accuracy: 0.902344
step: 1200, loss: 65.776703, accuracy: 0.925781
step: 1300, loss: 94.195862, accuracy: 0.910156
step: 1400, loss: 79.425507, accuracy: 0.917969
step: 1500, loss: 93.508163, accuracy: 0.914062
step: 1600, loss: 88.912506, accuracy: 0.917969
step: 1700, loss: 79.033607, accuracy: 0.929688
step: 1800, loss: 65.788315, accuracy: 0.898438
step: 1900, loss: 73.462387, accuracy: 0.937500
step: 2000, loss: 59.309540, accuracy: 0.917969
step: 2100, loss: 67.014008, accuracy: 0.917969
step: 2200, loss: 48.297115, accuracy: 0.949219
step: 2300, loss: 64.523148, accuracy: 0.910156
step: 2400, loss: 72.989517, accuracy: 0.925781
step: 2500, loss: 57.588585, accuracy: 0.929688
step: 2600, loss: 44.957100, accuracy: 0.960938
step: 2700, loss: 59.788242, accuracy: 0.937500
step: 2800, loss: 63.581337, accuracy: 0.937500
step: 2900, loss: 53.471252, accuracy: 0.941406
step: 3000, loss: 43.869728, accuracy: 0.949219
```

```python
# 在验证集上测试模型
pred = neural_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))
```

```python
# 可视化预测
import matplotlib.pyplot as plt

# 从验证集中预测5张图像
n_images = 5
test_images = x_test[:n_images]
predictions = neural_net(test_images)

# 显示图片和模型预测结果
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))
```

**output:**

![](http://qiniu.aihubs.net/blog/20190918/sUValrKo0OUz.png?imageslim)
Model prediction: 7

![](http://qiniu.aihubs.net/blog/20190918/yoLePpygnqiB.png?imageslim)
Model prediction:2

![](http://qiniu.aihubs.net/blog/20190918/IDwp4Wt4TzeD.png?imageslim)
Model prediction: 1

![](http://qiniu.aihubs.net/blog/20190918/bPuzTiXOwtLJ.png?imageslim)
Model prediction: 0

![](http://qiniu.aihubs.net/blog/20190918/89jwopvf1t50.png?imageslim)
Model prediction: 4



