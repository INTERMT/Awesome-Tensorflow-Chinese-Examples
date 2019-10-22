# Hello World
使用TensorFlow v2张量的一个简单的“hello world”示例

- 作者: Aymeric Damien
- 原项目: https://github.com/aymericdamien/TensorFlow-Examples/

```python
import tensorflow as tf
# 创建一个张量
hello = tf.constant("hello world")
print hello
```
**Output:**
```
tf.Tensor(hello world, shape=(), dtype=string)
```

```python
 # 访问张量的值，调用numpy()
print hello.numpy()
```

**output:**
```
hello world
```
