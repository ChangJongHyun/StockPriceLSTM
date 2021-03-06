import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tools import fetch_cosine_values, format_dataset, fetch_stock_price
from evaluate_ts import evaluate_ts

tf.reset_default_graph()
tf.set_random_seed(101)

feat_dimension = 20
train_size = 250
test_size = 250

# 1. define some parameters for Tensorflow
learning_rate = 0.01
optimizer = tf.train.AdamOptimizer
n_epochs = 100

# 2. prepare the observation matrics, for training and testing
stock_price = fetch_stock_price("FB", datetime.date(2015, 1, 1), datetime.date(2017, 12, 13))
cos_values = fetch_cosine_values(train_size + test_size + feat_dimension)
minibatch_cos_x, minibatch_cos_y = format_dataset(cos_values, feat_dimension)

train_x = minibatch_cos_x[:train_size, :].astype(np.float32)
train_y = minibatch_cos_y[:train_size].reshape((-1, 1)).astype(np.float32)
test_x = minibatch_cos_x[train_size:, :].astype(np.float32)
test_y = minibatch_cos_y[train_size:].reshape((-1, 1)).astype(np.float32)

# define placeholder for observation matrix and the labels
x_tf = tf.placeholder("float", shape=(None, feat_dimension), name="X")
y_tf = tf.placeholder("float", shape=(None, 1), name="Y")


# regression
def regression_ANN(x, weights, biases):
    return tf.add(biases, tf.matmul(x, weights))


weights = tf.Variable(tf.truncated_normal([feat_dimension, 1], mean=0.0, stddev=1.0), name="weights")
biases = tf.Variable(tf.zeros([1, 1]), name="bias")

y_pred = regression_ANN(x_tf, weights, biases)
cost = tf.reduce_mean(tf.square(y_tf - y_pred))  # MSE
train_op = optimizer(learning_rate=learning_rate).minimize(cost)

# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # For each epoch, the whole training set is feeded into the tensorflow graph
    for i in range(n_epochs):
        train_cost, _ = sess.run([cost, train_op], feed_dict={x_tf: train_x, y_tf: train_y})
        print("Training iteration #", i, "\nMSE: ", train_cost)

    test_cost, y_pr = sess.run([cost, y_pred], feed_dict={x_tf: test_x, y_tf: test_y})
    print("Test dataset: ", test_cost)

    # Evaluate the results
    evaluate_ts(test_x, test_y, y_pr)

    plt.plot(range(len(cos_values)), cos_values, 'b')
    plt.plot(range(len(cos_values) - test_size, len(cos_values)), y_pr, 'r--')
    plt.xlabel("Days")
    plt.ylabel("Predicted and true values")
    plt.title("Predicted (Red) VS Real (Blue)")
    plt.show()