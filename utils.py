import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from distill_optimizer import DistillOptimizer

config = json.load(open("config.json"))

def set_random_seed():
    np.random.seed(config["seed"])
    tf.set_random_seed(config["seed"])
    random.seed(config["seed"])

def get_session(graph):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config, graph=graph)
    return session

def add_to_collection(names, vals):
    for name, val in zip(names, vals):
        tf.add_to_collection(name, val)

def get_collection(names, graph):
    return (graph.get_collection(name)[0] for name in names)

def get_momentum(t):
    if t < config["momentum_saturation_time"]:
        return (config["initial_momentum"] * (1 - t / config["momentum_saturation_time"])
                + config["final_momentum"] * t / config["momentum_saturation_time"])
    else:
        return config["final_momentum"]

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

def get_optimizer(name):
    if name == "Momentum":
        return tf.train.MomentumOptimizer
    elif name == "DistillOptimizer":
        return DistillOptimizer

def save_data(train, test, model_name):
    return (pd.DataFrame({"train": train, "test": test})
            .to_csv("accuracy_{}.csv".format(model_name)))
