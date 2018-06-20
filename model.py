import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from distill_optimizer import DistillOptimizer
import utils as U

class Model(object):

    def __init__(self, network_type):
        self.graph = tf.Graph()
        self.session = U.get_session(self.graph)
        self.config = json.load(open("config.json"))
        self.network_type = network_type
        self.get_data()
        self.create_placeholders()
        self.build_model()

    def get_data(self):
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=False, seed=self.config["seed"])
        self.num_itr = self.mnist.train.num_examples // self.config["batch_size"]

    def create_placeholders(self):
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, shape=(None, self.config["input_dim"]), name="x")
            self.y = tf.placeholder(tf.int32, shape=(None,), name="y")
            self.probs_placeholder = tf.placeholder(tf.float32, shape=(None, self.config["output_dim"]), name="probs")
            self.keep_prob_visible_unit = tf.placeholder_with_default(1.0, shape=None, name="keep_prob_visible_unit")
            self.keep_prob_hidden_unit = tf.placeholder_with_default(1.0, shape=None, name="keep_prob_hidden_unit")
            self.learning_rate = tf.placeholder(tf.float32, shape=None, name="learning_rate")
            self.momentum = tf.placeholder(tf.float32, shape=None, name="momentum")
            self.max_norm = tf.placeholder(tf.float32, shape=None, name="max_norm")
            self.temperature_ph = tf.placeholder(tf.float32, shape=None, name="temperature")

    def model_architecture(self, num_hidden_units, use_probs):
        with self.graph.as_default():
            U.set_random_seed()
            w1 = tf.Variable(tf.random_normal(shape=(self.config["input_dim"], num_hidden_units), mean=0.0, stddev=0.01), name="w1")
            b1 = tf.Variable(tf.zeros(shape=(num_hidden_units,), dtype=tf.float32), name="b1")
            w2 = tf.Variable(tf.random_normal(shape=(num_hidden_units, num_hidden_units), mean=0.0, stddev=0.01), name="w2")
            b2 = tf.Variable(tf.zeros(shape=(num_hidden_units,), dtype=tf.float32), name="b2")
            w3 = tf.Variable(tf.random_normal(shape=(num_hidden_units, self.config["output_dim"]), mean=0.0, stddev=0.01), name="w3")
            b3 = tf.Variable(tf.zeros(shape=(self.config["output_dim"],), dtype=tf.float32), name="b3")

            x_dropout = tf.nn.dropout(self.x, self.keep_prob_visible_unit)
            h1 = tf.nn.relu(tf.matmul(x_dropout, w1) + b1)
            h1_dropout = tf.nn.dropout(h1, self.keep_prob_hidden_unit)
            h2 = tf.nn.relu(tf.matmul(h1_dropout, w2) + b2)
            h2_dropout = tf.nn.dropout(h2, self.keep_prob_hidden_unit)
            self.logits = tf.matmul(h2_dropout, w3) + b3
            self.probs = tf.nn.softmax(self.logits / self.temperature_ph)
            if use_probs:
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.probs_placeholder,
                                                                                      logits=self.logits / self.temperature_ph))
            else:
                self.loss = tf.reduce_mean(
                          tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
            self.pred_ = tf.argmax(self.logits, axis=1, output_type=tf.int32)
            self.missclassification_error = tf.reduce_sum(tf.cast(tf.not_equal(self.pred_, self.y), tf.float32))
            self.train_op = self.get_optimizer().minimize(self.loss)
            self.initialize = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def get_optimizer(self):
        optimizer_name = self.config[self.network_type]["optimizer"]
        if  optimizer_name == "DistillOptimizer":
            return U.get_optimizer(optimizer_name)(self.learning_rate, self.momentum, self.max_norm)
        elif optimizer_name == "Momentum":
            return U.get_optimizer(optimizer_name)(self.learning_rate, self.momentum)

    def build_model(self):
        self.model_architecture(self.config[self.network_type]["num_hidden_units"],
                                self.config[self.network_type]["use_probs"])

    def train(self):
        self.training_miss_classifications = []
        self.testing_miss_classifications = []
        self.session.run(self.initialize)
        lr = self.config[self.network_type]["initial_learning_rate"]
        x_test, y_test = self.mnist.test.images, self.mnist.test.labels
        for epoch in range(self.config["num_epoch"]):
            miss_classes = []
            mu = U.get_momentum(epoch)
            for itr in range(self.num_itr):
                x_train, y_train = self.mnist.train.next_batch(self.config["batch_size"])
                if self.config[self.network_type]["jitter_images"]:
                    x_train = U.jitter_images(x_train)
                _, miss_class, logits_val = self.session.run([self.train_op, self.missclassification_error, self.logits],
                                              feed_dict={self.x: x_train,
                                                         self.y: y_train,
                                                         self.keep_prob_hidden_unit: self.config[self.network_type]["keep_prob_hidden_unit"],
                                                         self.keep_prob_visible_unit: self.config[self.network_type]["keep_prob_visible_unit"],
                                                         self.learning_rate: lr,
                                                         self.momentum: mu,
                                                         self.max_norm: self.config["max_norm_val"]})
                miss_classes.append(miss_class)
            lr *= self.config["learning_rate_decay"]
            if (epoch + 1) % self.config["show_every"] == 0:
                test_miss_class = self.session.run(self.missclassification_error,
                                           feed_dict={self.x: x_test,
                                                      self.y: y_test,
                                                      self.keep_prob_hidden_unit: 1.0,
                                                      self.keep_prob_visible_unit: 1.0})
                print("epoc: {0}, train_miss_class: {1:0.0f}, test_miss_class: {2:0.0f}"
                      .format(epoch, np.sum(miss_classes), test_miss_class))
                self.training_miss_classifications.append(np.sum(miss_classes))
                self.testing_miss_classifications.append(test_miss_class)
        U.save_data(self.training_miss_classifications, self.testing_miss_classifications, self.network_type)
        U.plot_results(self.training_miss_classifications, self.testing_miss_classifications, self.network_type)

    def save(self):
        names = ["probs", "logits", "inputs", "temperature_ph"]
        vals  =  [self.probs, self.logits, self.x, self.temperature_ph]
        with self.graph.as_default():
            U.add_to_collection(names, vals)
            self.saver.save(self.session, "checkpoint/" + self.network_type)

if __name__ == "__main__":
    model = Model("ensemble")
    model.train()
    model.save()







































#####
