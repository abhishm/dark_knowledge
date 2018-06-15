seed = 42

import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

num_epoch = 3000
batch_size = 128
initial_momentum = 0.5
final_momentum = 0.99
learning_rate_decay = 0.998
momentum_saturation_time = 500
max_norm_val = 15.
temperature = 10.

def get_momentum(t):
    if t < momentum_saturation_time:
        return (initial_momentum * (1 - t / momentum_saturation_time)
                + final_momentum * t / momentum_saturation_time)
    else:
        return final_momentum


def jitter_images(x):
    jitter = np.random.randint(3, size=(2))
    x = x.reshape(-1, 28, 28)
    x = np.roll(x, tuple(jitter), axis=(1, 2))
    return np.reshape(x, (-1, 28 * 28))


def plot_results(train, test, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
    axes[0].plot(train)
    axes[1].plot(test)
    axes[0].set_ylim([100, 500])
    axes[1].set_ylim([100, 500])
    axes[0].set_title("training missclassification")
    axes[1].set_title("testing missclassification")
    plt.savefig("{}_miss_classification_plot.png".format(model_name), dpi=700)

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
num_examples = mnist.train.num_examples
num_itr = num_examples // batch_size

# Create placeholders
x = tf.placeholder(tf.float32, shape=(None, 784), name="x")
y = tf.placeholder(tf.int32, shape=(None,), name="y")
probs_placeholder = tf.placeholder(tf.float32, shape=(None, 10), name="probs")
keep_prob_visible_unit = tf.placeholder(tf.float32, shape=None, name="keep_prob_visible_unit")
keep_prob_hidden_unit = tf.placeholder(tf.float32, shape=None, name="keep_prob_hidden_unit")
learning_rate = tf.placeholder(tf.float32, shape=None, name="learning_rate")
momentum = tf.placeholder(tf.float32, shape=None, name="momentum")
max_norm = tf.placeholder(tf.float32, shape=None, name="max_norm")
temperature_ph = tf.placeholder(tf.float32, shape=None, name="temperature")

def get_model(num_hidden_units, normalize_vars, use_probs):
    with tf.variable_scope("ensemble"):
        w1 = tf.Variable(tf.random_normal(shape=(784, num_hidden_units), mean=0.0, stddev=0.01, seed=seed), name="w1")
        delta_w1 = tf.Variable(tf.zeros(shape=(784, num_hidden_units), dtype=tf.float32), name="delta_w1", trainable=False)
        b1 = tf.Variable(tf.zeros(shape=(num_hidden_units,), dtype=tf.float32), name="b1")
        delta_b1 = tf.Variable(tf.zeros(shape=(num_hidden_units,), dtype=tf.float32), name="delta_b1", trainable=False)
        w2 = tf.Variable(tf.random_normal(shape=(num_hidden_units, num_hidden_units), mean=0.0, stddev=0.01, seed=seed), name="w2")
        delta_w2 = tf.Variable(tf.zeros(shape=(num_hidden_units, num_hidden_units), dtype=tf.float32), name="delta_w2", trainable=False)
        b2 = tf.Variable(tf.zeros(shape=(num_hidden_units,), dtype=tf.float32), name="b2")
        delta_b2 = tf.Variable(tf.zeros(shape=(num_hidden_units,), dtype=tf.float32), name="delta_b2", trainable=False)
        w3 = tf.Variable(tf.random_normal(shape=(num_hidden_units, 10), mean=0.0, stddev=0.01, seed=seed), name="w3")
        delta_w3 = tf.Variable(tf.zeros(shape=(num_hidden_units, 10), dtype=tf.float32), name="delta_w3", trainable=False)
        b3 = tf.Variable(tf.zeros(shape=(10,), dtype=tf.float32), name="b3")
        delta_b3 = tf.Variable(tf.zeros(shape=(10,), dtype=tf.float32), name="delta_b3", trainable=False)
        x_dropout = tf.nn.dropout(x, keep_prob_visible_unit)
        h1 = tf.nn.relu(tf.matmul(x_dropout, w1) + b1)
        h1_dropout = tf.nn.dropout(h1, keep_prob_hidden_unit)
        h2 = tf.nn.relu(tf.matmul(h1_dropout, w2) + b2)
        h2_dropout = tf.nn.dropout(h2, keep_prob_hidden_unit)
        logits = tf.matmul(h2_dropout, w3) + b3
        if use_probs:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=probs_placeholder,
                                                                             logits=logits / temperature))
        else:
            loss = tf.reduce_mean(
                      tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
        pred_ = tf.argmax(logits, axis=1, output_type=tf.int32)
        missclassification_error = tf.reduce_sum(tf.cast(tf.not_equal(pred_, y), tf.float32))
        vars_ = [w1, b1, w2, b2, w3, b3]
        delta_vars = [delta_w1, delta_b1, delta_w2, delta_b2, delta_w3, delta_b3]
        grads = tf.gradients(loss, vars_)
        assign_op_vars = []
        for var, delta_var, grad in zip(vars_, delta_vars, grads):
            assign_op_delta_var = delta_var.assign(momentum * delta_var - (1 - momentum) * learning_rate * grad)
            with tf.control_dependencies([assign_op_delta_var]):
                if normalize_vars:
                    assign_op_var = var.assign(tf.clip_by_norm(var + delta_var, max_norm, axes=[0]))
                else:
                    assign_op_var = var.assign(var + delta_var)
            assign_op_vars.append(assign_op_var)
        return train_op, missclassification_error, logits

def get_params(model_type):
    if model_type == "ensemble":
        return {"keep_prob_hidden_unit": 0.5,
                "keep_prob_visible_unit": 0.8,
                "jitter_images": True,
                "num_hidden_units": 1200,
                "normalize_vars": True,
                "initial_learning_rate": 1}
    elif model_type == "small":
        return {"keep_prob_hidden_unit": 1.0,
                "keep_prob_visible_unit": 1.0,
                "jitter_images": False,
                "num_hidden_units": 800,
                "normalize_vars":False,
                "initial_learning_rate": 1}
    elif model_type == "distill":
        return {"keep_prob_hidden_unit": 1.0,
                "keep_prob_visible_unit": 1.0,
                "jitter_images": False,
                "num_hidden_units": 800,
                "normalize_vars":False,
                "initial_learning_rate": 2}

def train_network(network_type, sess):
    x_test, y_test = mnist.test.images, mnist.test.labels
    training_miss_classifincations = []
    testing_miss_classifications = []
    training_params = get_params(network_type)
    training_ops = get_model(training_params["num_hidden_units"],
                             training_params["normalize_vars"],
                             use_probs=False)
    train_op, missclassification_error, logits = training_ops
    sess.run(tf.global_variables_initializer())
    lr = training_params["initial_learning_rate"]
    for epoch in range(num_epoch):
        miss_classes = []
        mu = get_momentum(epoch)
        for itr in range(num_itr):
            x_train, y_train = mnist.train.next_batch(batch_size)
            if training_params["jitter_images"]:
                x_train = jitter_images(x_train)
            _, miss_class, logits_val = sess.run(training_ops,
                                          feed_dict={x: x_train,
                                                     y: y_train,
                                                     keep_prob_hidden_unit: training_params["keep_prob_hidden_unit"],
                                                     keep_prob_visible_unit: training_params["keep_prob_visible_unit"],
                                                     learning_rate: lr,
                                                     momentum: mu,
                                                     max_norm: max_norm_val})
            miss_classes.append(miss_class)
        lr *= learning_rate_decay
        test_miss_class = sess.run(missclassification_error,
                                   feed_dict={x: x_test,
                                             y: y_test,
                                             keep_prob_hidden_unit: 1.0,
                                             keep_prob_visible_unit: 1.0})
        print("epoc: {0}, train_miss_class: {1:0.0f}, test_miss_class: {2:0.0f}"
              .format(epoch, np.sum(miss_classes), test_miss_class))
        training_miss_classifincations.append(np.sum(miss_classes))
        testing_miss_classifications.append(test_miss_class)
    plot_results(training_miss_classifincations, testing_miss_classifications, network_type)
    return tf.nn.softmax(logits / temperature_ph)

def distill_knowledge():
    def get_probs(batch):
        return sess.run(ensemble_probs, {x: x_train,
                                         keep_prob_hidden_unit: 1.0,
                                         keep_prob_visible_unit: 1.0,
                                         temperature_ph: 1.0})
    x_test, y_test = mnist.test.images, mnist.test.labels
    training_miss_classifincations = []
    testing_miss_classifications = []

    sess = get_session()
    with tf.variable_scope("ensemble"):
        ensemble_probs = train_network("ensemble", sess)
    training_params = get_params("distill")
    lr = training_params["initial_learning_rate"]
    with tf.variable_scope("distill"):
        training_ops = get_model(training_params["num_hidden_units"],
                                 training_params["normalize_vars"],
                                 use_probs=True)
        train_op, missclassification_error, logits = training_ops

    sess.run(tf.variables_initializer(tf.get_collection("variables", scope="distill")))
    for epoch in range(num_epoch):
        miss_classes = []
        mu = get_momentum(epoch)
        for itr in range(num_itr):
            x_train, y_train = mnist.train.next_batch(batch_size)
            if training_params["jitter_images"]:
                x_train = jitter_images(x_train)
            _, miss_class, logits = sess.run(training_ops,
                                          feed_dict={x: x_train,
                                                     y: y_train,
                                                     probs_placeholder: get_probs(x_train),
                                                     keep_prob_hidden_unit: training_params["keep_prob_hidden_unit"],
                                                     keep_prob_visible_unit: training_params["keep_prob_visible_unit"],
                                                     temperature_ph: temperature,
                                                     learning_rate: lr,
                                                     momentum: mu,
                                                     max_norm: max_norm_val})
            miss_classes.append(miss_class)
        lr *= learning_rate_decay
        test_miss_class = sess.run(missclassification_error,
                                   feed_dict={x: x_test,
                                             y: y_test,
                                             temperature_ph: temperature,
                                             keep_prob_hidden_unit: 1.0,
                                             keep_prob_visible_unit: 1.0})
        print("epoc: {0}, train_miss_class: {1:0.0f}, test_miss_class: {2:0.0f}"
              .format(epoch, np.sum(miss_classes), test_miss_class))
        training_miss_classifincations.append(np.sum(miss_classes))
        testing_miss_classifications.append(test_miss_class)
    plot_results(training_miss_classifincations, testing_miss_classifications, "distill")



if __name__ == "__main__":
    import sys
    sess = get_session()
    # def get_probs(batch):
    #     return sess.run(ensemble_probs, {x: batch})
    #
    # x_test, y_test = mnist.test.images, mnist.test.labels
    # num_epoch = 100
    # with tf.variable_scope("ensemble"):
    #     ensemble_probs = train_network("ensemble", sess)
    # feed_dict={x: x_test,
    #           keep_prob_hidden_unit: 1.0,
    #           keep_prob_visible_unit: 1.0,
    #           temperature_ph: 1.0}
    # p_ensemble = sess.run(ensemble_probs, feed_dict)

    distill_knowledge()
    # train_network(sys.argv[1], sess)






































#####
