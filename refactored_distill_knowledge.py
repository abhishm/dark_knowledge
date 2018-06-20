from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import Model
import utils as U

class DistillModel(Model):
    def __init__(self, network_type):
        super().__init__(network_type)
        self.restore_ensemble_model()

    def restore_ensemble_model(self):
        self.ensemble_graph = tf.Graph()
        self.ensemble_session = U.get_session(self.ensemble_graph)
        with self.ensemble_graph.as_default():
            saver = tf.train.import_meta_graph("checkpoint/ensemble.meta")
            saver.restore(self.ensemble_session, "checkpoint/ensemble")
        names = ["probs", "logits", "temperature_ph", "inputs"]
        self.ensemble_model = dict(zip(names, U.get_collection(names, self.ensemble_graph)))

    def get_probs(self, batch):
        return self.ensemble_session.run(self.ensemble_model["probs"], {self.ensemble_model["inputs"]: batch,
                                                                 self.ensemble_model["temperature_ph"]: self.config["temperature"]})

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
                _, miss_class, logits_val = self.session.run([self.train_op, self.missclassification_error, self.logits],
                                              feed_dict={self.x: x_train,
                                                         self.y: y_train,
                                                         self.probs_placeholder: self.get_probs(x_train),
                                                         self.keep_prob_hidden_unit: self.config[self.network_type]["keep_prob_hidden_unit"],
                                                         self.keep_prob_visible_unit: self.config[self.network_type]["keep_prob_visible_unit"],
                                                         self.temperature_ph: self.config["temperature"],
                                                         self.learning_rate: lr,
                                                         self.momentum: mu,
                                                         self.max_norm: self.config["max_norm_val"]})
                miss_classes.append(miss_class)
            lr *= self.config["learning_rate_decay"]
            test_miss_class = self.session.run(self.missclassification_error,
                                       feed_dict={self.x: x_test,
                                                  self.y: y_test,
                                                  self.keep_prob_hidden_unit: 1.0,
                                                  self.keep_prob_visible_unit: 1.0})
            if (epoch + 1) % self.config["show_every"] == 0:
                print("epoc: {0}, train_miss_class: {1:0.0f}, test_miss_class: {2:0.0f}"
                      .format(epoch, np.sum(miss_classes), test_miss_class))
                self.training_miss_classifications.append(np.sum(miss_classes))
                self.testing_miss_classifications.append(test_miss_class)
        U.save_data(self.training_miss_classifications, self.testing_miss_classifications)
        U.plot_results(self.training_miss_classifications, self.testing_miss_classifications, self.network_type)

if __name__ == "__main__":
    model = DistillModel("distill")
    model.train()
    model.save()
