import numpy as np
import re
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
# 2019.4.14

path = 'housing.data'
fr = open(path)
lines = fr.readlines()
dataset = []
for line in lines:
    line = line.strip()
    line = re.split(r' +', line)
    line = [float(i) for i in line]
    dataset.append(line)

# data handle
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
dataset_array = np.array(dataset)
dataset_pd = pd.DataFrame(dataset_array, columns=housing_header)
housing_data = dataset_pd.get(cols_used)
labels = dataset_pd.get('MEDV')
housing_data = np.array(housing_data)
# data scaling
housing_data = (housing_data - housing_data.min(0))/housing_data.ptp(0)
labels = np.array(labels)[:, np.newaxis]

# test and train data split
np.random.seed(13)
train_ratio = 0.8
data_size = len(housing_data)
train_index = np.random.choice(a=data_size, size=round(data_size*train_ratio), replace=False)
test_index = np.array(list(set(range(data_size))-set(train_index)))

x_train = housing_data[train_index]
y_train = labels[train_index]
x_test = housing_data[test_index]
y_test = labels[test_index]
# parameters
num_features = len(cols_used)
batchsize = len(x_test)
K = 4
# placeholder
x_train_placeholder = tf.placeholder(tf.float32, [None, num_features])
x_test_placeholder = tf.placeholder(tf.float32, [None, num_features])

y_train_placeholder = tf.placeholder(tf.float32, [None, 1])
y_test_placeholder = tf.placeholder(tf.float32, [None, 1])

# distance
distance = tf.reduce_sum(tf.abs(tf.subtract(x_train_placeholder, tf.expand_dims(x_test_placeholder, 1))), axis=2)

# distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_train_placeholder, tf.expand_dims(x_test_placeholder, 1))), reduction_indices=[1]))
# x_test_placeholder_expansion = tf.expand_dims(x_test_placeholder, 1)
# x_substract = x_train_placeholder - x_test_placeholder_expansion
# x_square = tf.square(x_substract)
# x_square_sum = tf.reduce_sum(x_square, reduction_indices=[1])
# x_sqrt = tf.sqrt(x_square_sum)

# rank to predict
top_k_value, top_k_index = tf.nn.top_k(tf.negative(distance), k=K)

top_k_value = tf.truediv(1.0, top_k_value)

top_k_value_sum = tf.reduce_sum(top_k_value, axis=1)
top_k_value_sum = tf.expand_dims(top_k_value_sum, 1)
top_k_value_sum_again = tf.matmul(top_k_value_sum, tf.ones([1, K], dtype=tf.float32))
# get weights
top_k_weights = tf.div(top_k_value, top_k_value_sum_again)

weights = tf.expand_dims(top_k_weights, 1)
# get top k labels
top_k_y = tf.gather(y_train_placeholder, top_k_index)
# predict
predictions = tf.squeeze(tf.matmul(weights, top_k_y), axis=[1])

# loss function
loss = tf.reduce_mean(tf.square(tf.subtract(predictions, y_test_placeholder)))

loop_nums = int(np.ceil(len(x_test)/batchsize))

with tf.Session() as sess:
    for i in range(loop_nums):
        # batchsize set
        min_index = i*batchsize
        max_index = min((i+1)*batchsize, len(x_test))
        x_test_batch = x_test[min_index: max_index]
        y_test_batch = y_test[min_index: max_index]
        # run
        result, los = sess.run([predictions, loss], feed_dict={
            x_train_placeholder: x_train, y_train_placeholder: y_train,
            x_test_placeholder: x_test_batch, y_test_placeholder: y_test_batch
        })

        print("No.%d batch, loss is %f"%(i+1, los))
# plot
pins = np.linspace(5, 50, 45)
plt.hist(result, pins, alpha=0.5, label='prediction')
plt.hist(y_test_batch, pins, alpha=0.5, label='actual')
plt.legend(loc='best')
plt.show()


