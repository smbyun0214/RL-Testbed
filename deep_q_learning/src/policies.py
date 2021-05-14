import numpy as np
import tensorflow as tf


def epsilon_greedy_policy(model, state, epsilon):
    _, num_actions = model.output_shape
    sample = np.random.sample()
    if sample < epsilon:
        return np.random.choice(num_actions)
    state = np.expand_dims(state, axis=0)
    with tf.device('/CPU:0'):
        return np.argmax(model(state), axis=1)[0]
