import random
import numpy as np
import pandas as pd
import tensorflow as tf
from distill_optimizer import DistillOptimizer

seed = 42
def set_random_seed():
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

input_size = 10
lr = 0.1
mu = 0.9
max_norm = 15.

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def exp_1():
    print("Experiment 1")
    x = tf.placeholder(tf.float32, (None, input_size), name="x")
    w1 = tf.Variable(tf.random_normal((input_size, 1), stddev=0.001), name="w2")
    w1_delta = tf.Variable(tf.zeros((input_size, 1)), trainable=False)
    w_ = np.random.random((input_size, 1)).astype(np.float32)
    y = tf.matmul(x, w_)
    output = tf.matmul(x, w1)
    losses = []

    loss = tf.reduce_mean((output - y) ** 2)
    grads = tf.gradients(loss, [w1])
    update_w1_delta = w1_delta.assign(mu * w1_delta - (1- mu) * lr * grads[0])
    with tf.control_dependencies([update_w1_delta]):
        train_op = w1.assign(tf.clip_by_norm(w1 + w1_delta, max_norm, axes=0))

    sess = get_session()
    sess.run(tf.global_variables_initializer())
    for _ in range(num_itr):
        x_ = np.random.rand(10, input_size)
        _, loss_ = sess.run([train_op, loss], {x: x_})
        losses.append(loss_)
    return losses

def exp_2():
    print("Experiment 2")
    x = tf.placeholder(tf.float32, (None, input_size), name="x")
    w1 = tf.Variable(tf.random_normal((input_size, 1), stddev=0.001), name="w2")
    w1_delta = tf.Variable(tf.zeros((input_size, 1)), trainable=False)
    w_ = np.random.random((input_size, 1)).astype(np.float32)
    y = tf.matmul(x, w_)
    output = tf.matmul(x, w1)
    losses = []

    loss = tf.reduce_mean((output - y) ** 2)
    optimizer = DistillOptimizer(learning_rate=lr, mu=mu, max_norm=max_norm)

    train_op = optimizer.minimize(loss)

    sess = get_session()
    sess.run(tf.global_variables_initializer())
    for _ in range(num_itr):
        x_ = np.random.rand(10, input_size)
        _, loss_ = sess.run([train_op, loss], {x: x_})
        losses.append(loss_)
    return losses

def combine_two_experiemnts():
    set_random_seed()
    loss_1 = exp_1()
    tf.reset_default_graph()
    set_random_seed()
    loss_2 = exp_2()
    return pd.DataFrame({"low_api_implementation": loss_1,
                         "high_api_inplementation": loss_2})

if __name__ == "__main__":
    import sys
    try:
        num_itr = int(sys.argv[1])
    except:
        num_itr = 100
    print(combine_two_experiemnts()[-20:])
