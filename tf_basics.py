import os
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)

# Initialization of tensors

# x1 = tf.constant(4, shape=(1, 1), dtype=tf.float32)
# print("x1 = ", x1)
# x2 = tf.constant([[1, 2, 3], [4, 5, 6]])
# print("x2 =  ", x2)
# a = tf.ones((3, 3))
# print("a = ", a)
# b = tf.zeros((2, 3))
# print("b = ", b)
# c = tf.eye(3) # eye(I) for identity matrix
# print("c = ", c)
# d = tf.random.normal((3, 3), mean=0, stddev=1)
# print("d = ", d)
# e = tf.random.uniform((1, 3), minval=0, maxval=1)
# print("e = ", e)
# f = tf.range(start=1, limit=10, delta=2)
# print("f = ", f)

# Mathematical Operations

# x = tf.constant([1, 2, 3])
# y = tf.constant([9, 8, 7])
#
# print("tf_Add = ", tf.add(x, y))
# print("Add = ", x + y)
#
# print("tf_subtract = ", tf.subtract(x, y))
# print("tf_divide = ", tf.divide(x, y))
# print("tf_multiply = ", tf.multiply(x, y))
#
# print("tensordot = ", tf.tensordot(x, y, axes=1))
# print("tensordot_alternative = ", tf.reduce_sum(x*y, axis=0))
#
# x = tf.random.normal((2, 3))
# y = tf.random.normal((3, 4))
#
# z = tf.matmul(x, y)
# print(z)
# z = x@y
# print(z)

# Indexing of tensors

# x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
# print(x[:])
# print(x[1:])
# print(x[1:3])
# print(x[::2])
# indices = tf.constant([0, 3])
# x_ind = tf.gather(x, indices)
# print((x_ind))

# x = tf.constant([[1, 2],
#                  [3, 4],
#                  [5, 6]])
#
# print(x[0, :])
# print(x[0:2, :])

# Reshaping


x = tf.range(9)
print(x)

x = tf.reshape(x, (3, 3))
print(x)

x = tf.transpose(x, perm=[1, 0])
print(x)
