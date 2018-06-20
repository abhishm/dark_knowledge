import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from distill_optimizer import DistillOptimizer

seed = 42
num_epoch = 3
batch_size = 128
initial_momentum = 0.5
final_momentum = 0.99
learning_rate_decay = 0.998
momentum_saturation_time = 500
max_norm_val = 15.
temperature = 10.

# Dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
num_examples = mnist.train.num_examples
num_itr = num_examples // batch_size
x_test, y_test = mnist.test.images, mnist.test.labels

# Create placeholders
x = tf.placeholder(tf.float32, shape=(None, 784), name="x")
y = tf.placeholder(tf.int32, shape=(None,), name="y")
probs_placeholder = tf.placeholder(tf.float32, shape=(None, 10), name="probs")
keep_prob_visible_unit = tf.placeholder_with_default(1.0, shape=None, name="keep_prob_visible_unit")
keep_prob_hidden_unit = tf.placeholder_with_default(1.0, shape=None, name="keep_prob_hidden_unit")
learning_rate = tf.placeholder(tf.float32, shape=None, name="learning_rate")
momentum = tf.placeholder(tf.float32, shape=None, name="momentum")
max_norm = tf.placeholder(tf.float32, shape=None, name="max_norm")
temperature_ph = tf.placeholder(tf.float32, shape=None, name="temperature")


def set_random_seed():
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def add_to_collection(names, vals):
    for name, val in zip(names, vals):
        tf.add_to_collection(name, val)

def get_momentum(t):
    if t < momentum_saturation_time:
        return (initial_momentum * (1 - t / momentum_saturation_time)
                + final_momentum * t / momentum_saturation_time)
    else:
        return final_momentum

def save_model(sess, model_name):
    saver = tf.train.Saver()
    saver.save(sess, "checkpoint/" + model_name)

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

def get_model(num_hidden_units, normalize_vars, use_probs, optimizer):
    w1 = tf.Variable(tf.random_normal(shape=(784, num_hidden_units), mean=0.0, stddev=0.01), name="w1")
    b1 = tf.Variable(tf.zeros(shape=(num_hidden_units,), dtype=tf.float32), name="b1")
    w2 = tf.Variable(tf.random_normal(shape=(num_hidden_units, num_hidden_units), mean=0.0, stddev=0.01), name="w2")
    b2 = tf.Variable(tf.zeros(shape=(num_hidden_units,), dtype=tf.float32), name="b2")
    w3 = tf.Variable(tf.random_normal(shape=(num_hidden_units, 10), mean=0.0, stddev=0.01), name="w3")
    b3 = tf.Variable(tf.zeros(shape=(10,), dtype=tf.float32), name="b3")

    x_dropout = tf.nn.dropout(x, keep_prob_visible_unit)
    h1 = tf.nn.relu(tf.matmul(x_dropout, w1) + b1)
    h1_dropout = tf.nn.dropout(h1, keep_prob_hidden_unit)
    h2 = tf.nn.relu(tf.matmul(h1_dropout, w2) + b2)
    h2_dropout = tf.nn.dropout(h2, keep_prob_hidden_unit)
    logits = tf.matmul(h2_dropout, w3) + b3

    if use_probs:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=probs_placeholder,
                                                                         logits=logits / temperature_ph))
    else:
        loss = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    pred_ = tf.argmax(logits, axis=1, output_type=tf.int32)
    missclassification_error = tf.reduce_sum(tf.cast(tf.not_equal(pred_, y), tf.float32))
    train_op = optimizer.minimize(loss)
    return train_op, missclassification_error, logits

def get_params(model_type):
    if model_type == "ensemble":
        return {"keep_prob_hidden_unit": 0.5,
                "keep_prob_visible_unit": 0.8,
                "jitter_images": True,
                "num_hidden_units": 1200,
                "normalize_vars": True,
                "initial_learning_rate": 1,
                "use_probs": False,
                "optimizer": DistillOptimizer(learning_rate, momentum, max_norm)}
    elif model_type == "small":
        return {"keep_prob_hidden_unit": 1.0,
                "keep_prob_visible_unit": 1.0,
                "jitter_images": False,
                "num_hidden_units": 800,
                "normalize_vars":False,
                "initial_learning_rate": 1,
                "use_probs": False,
                "optimizer": tf.train.MomentumOptimizer(learning_rate, momentum)}
    elif model_type == "distill":
        return {"keep_prob_hidden_unit": 1.0,
                "keep_prob_visible_unit": 1.0,
                "jitter_images": False,
                "num_hidden_units": 800,
                "normalize_vars":False,
                "initial_learning_rate": 2,
                "use_probs": True,
                "optimizer": tf.train.MomentumOptimizer(learning_rate, momentum)}

def train_network(network_type, sess):
    training_miss_classifincations = []
    testing_miss_classifications = []
    training_params = get_params(network_type)
    with tf.variable_scope(network_type):
        training_ops = get_model(training_params["num_hidden_units"],
                                 training_params["normalize_vars"],
                                 training_params["use_probs"],
                                 training_params["optimizer"])
        train_op, missclassification_error, logits = training_ops
    sess.run(tf.variables_initializer(tf.get_collection("variables", scope=network_type)))
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
    probs = tf.nn.softmax(logits / temperature_ph)
    if network_type == "ensemble":
        names = ["probs", "logits", "inputs", "temperature_ph"]
        vals =[probs, logits, x, temperature_ph]
        add_to_collection(names, vals)
        save_model(sess, network_type)
    return probs

def distill_knowledge():
    def get_probs(batch):
        return sess.run(ensemble_probs, {ensemble_inputs: x_train,
                                         ensemble_temperature_ph: temperature})

    training_miss_classifincations = []
    testing_miss_classifications = []

    sess = get_session()
    # Load an ensemble model if there is already a saved model
    if os.path.isfile("checkpoint/ensemble.meta"):
        saver = tf.train.import_meta_graph("checkpoint/ensemble.meta")
        saver.restore(sess, "checkpoint/ensemble")
        ensemble_probs = tf.get_collection("probs")[0]
        ensemble_inputs = tf.get_collection("inputs")[0]
        ensemble_temperature_ph = tf.get_collection("temperature_ph")[0]
    else: # train an ensemble model
        ensemble_probs = train_network("ensemble", sess)
        ensemble_inputs = x
        ensemble_temperature_ph = temperature_ph
    training_params = get_params("distill")
    lr = training_params["initial_learning_rate"]
    with tf.variable_scope("distill"):
        training_ops = get_model(training_params["num_hidden_units"],
                                 training_params["normalize_vars"],
                                 training_params["use_probs"],
                                 training_params["optimizer"])
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
                                              keep_prob_hidden_unit: 1.0,
                                              keep_prob_visible_unit: 1.0})
        print("epoc: {0}, train_miss_class: {1:0.0f}, test_miss_class: {2:0.0f}"
              .format(epoch, np.sum(miss_classes), test_miss_class))
        training_miss_classifincations.append(np.sum(miss_classes))
        testing_miss_classifications.append(test_miss_class)
    plot_results(training_miss_classifincations, testing_miss_classifications, "distill")



if __name__ == "__main__":
    import sys
    set_random_seed()
    sess = get_session()
    distill_knowledge()






































#####
