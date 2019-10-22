
# MNIST数据集[^1]介绍

大多数示例使用手写数字的MNIST数据集。该数据集包含60,000个用于训练的示例和10,000个用于测试的示例。这些数字已经过尺寸标准化并位于图像中心，图像是固定大小(28x28像素)，其值为0到1。为简单起见，每个图像都被平展并转换为784(28 * 28)个特征的一维numpy数组。

### 概览

![mark](http://qiniu.aihubs.net/blog/20190906/gcuRlsD0T2Et.png?imageslim)

### 用法

在我们的示例中，我们使用TensorFlow [input_data.py](https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/mnist/input_data.py)脚本来加载该数据集。

它对于管理我们的数据非常有用，并且可以处理：

- 加载数据集

- 将整个数据集加载到numpy数组中
```python
# 导入 MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 加载数据
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels
```

- `next_batch`函数，可以遍历整个数据集并仅返回所需的数据集样本部分(以节省内存并避免加载整个数据集)。  
```python
# 获取接下来的64个图像数组和标签
batch_X, batch_Y = mnist.train.next_batch(64)   
```
[^1]: http://yann.lecun.com/exdb/mnist/


