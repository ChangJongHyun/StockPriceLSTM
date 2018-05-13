import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from evaluate_ts import evaluate_ts
from tensorflow.contrib import rnn
from tools import fetch_cosine_values, format_dataset, fetch_stock_price

#  initialize graph and give random seed
tf.reset_default_graph()
tf.set_random_seed(101)

#  size of dataset
time_dimension = 20
train_size = 250
test_size = 250

#  hyper parameters
learning_rate = 0.01
optimizer = tf.train.AdagradOptimizer
n_epochs = 100
n_embeddings = 64

#  fetch noisy cosine and reshape it to have a 3D tensor shape
cos_values = fetch_cosine_values(train_size + test_size + time_dimension)
minibatch_cos_x, minibatch_cos_y = format_dataset(cos_values, time_dimension)
train_x = minibatch_cos_x[:train_size, :].astype(np.float32)
train_y = minibatch_cos_y[:train_size].reshape((-1, 1)).astype(np.float32)
test_x = minibatch_cos_x[:train_size:, :].astype(np.float32)
test_y = minibatch_cos_y[train_size:].reshape((-1, 1)).astype(np.float32)
train_x_ts = train_x[:, :, np.newaxis]
test_x_ts = test_x[:, :, np.newaxis]

x_tf = tf.placeholder("float", shape=(None, time_dimension, 1), name="X")
y_tf = tf.placeholder("float", shape=(None, 1), name="Y")


#  define the model(RNN) - LSTM(Long short term memory)
def RNN(x, weights, biases):
    x_ = tf.unstack(x, time_dimension, 1)
    lstm_cell = rnn.BasicLSTMCell(n_embeddings)
    outputs, _ = rnn.static_rnn(lstm_cell, x_, dtype=tf.float32)
    return tf.add(biases, tf.matmul(outputs[-1], weights))


#  set trainable variables, cost function and training operator
weights = tf.Variable(tf.truncated_normal([n_embeddings, 1], mean=0.0, stddev=1.0),
                      name="weights")
biases = tf.Variable(tf.zeros([1]), name="bias")
y_pred = RNN(x_tf, weights, biases)
cost = tf.reduce_mean(tf.square(y_tf - y_pred))
train_op = optimizer(learning_rate=learning_rate).minimize(cost)

#  session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(n_epochs):
        train_cost, _ = sess.run([cost, train_op], feed_dict={x_tf: train_x_ts, y_tf: train_y})
        if i % 100 is 0:
            print("Training iteration", i, "MSE", train_cost)

    #  check the performance on the test set
    test_cost, y_pr = sess.run([cost, y_pred], feed_dict={x_tf: test_x_ts, y_tf: test_y})
    print("Test dataset: ", test_cost)
    evaluate_ts(test_x, test_y, y_pr)

    #  plot prediction
    plt.plot(range(len(cos_values)), cos_values, 'b')
    plt.plot(range(len(cos_values) - test_size, len(cos_values)), y_pr, 'r--')
    plt.xlabel("Days")
    plt.ylabel("Predicted and true values")
    plt.title("Predicted (Red) VS Real (Blue)")
    plt.show()
