import tensorflow as tf

from collections import deque
from utils import MovingAverage
from models import Embedding, RND

class Episodic():
    def __init__(
        self, num_of_actions, epsilon=0.0001, num_of_neighbours=10, cluster_distance=0.008,
        pseudo_counts=0.001, maximum_similarity=8, memory_capacity=30000):
        self.epsilon = epsilon
        self.num_of_neighbours = num_of_neighbours
        self.cluster_distance = cluster_distance
        self.pseudo_counts = pseudo_counts
        self.maximum_similarity = maximum_similarity

        self.buffer = deque([], maxlen=memory_capacity)
        self.moving_average = MovingAverage()
        self.network = Embedding(num_of_actions)


    def reset(self):
        self.buffer.clear()
        self.moving_average.reset()


    def __call__(self, observations, next_observations):
        controllable_state = self.network(observations, next_observations, controllable_state=True)

        dists = [ tf.norm(controllable_state - state) for state in self.buffer ]
        dists = tf.nn.top_k(dists, self.num_of_neighbours)
        squared_dists = tf.math.square(dists)

        for squared_dist in squared_dists:
            self.moving_average(squared_dist)
        
        normalized_dists = squared_dists / self.moving_average.prev_mean
        normalized_dists = tf.math.maximum(normalized_dists - self.cluster_distance, 0.0)
        
        kernels = self.epsilon / (normalized_dists + self.epsilon)

        similarity = tf.math.sqrt(tf.math.reduce_sum(kernels)) + self.pseudo_counts

        self.buffer.append(controllable_state)
        
        if similarity > self.maximum_similarity:
            return 0.0
        else:
            return 1 / similarity


class LifeLong():
    def __init__(self):
        self.random_network = RND()
        self.predict_network = RND()
        self.moving_average = MovingAverage()
        
        self.random_network.trainable = False

    def reset(self):
        self.moving_average.reset()

    def __call__(self, observations):
        random = self.random_network(observations)
        predict = self.predict_network(observations)

        error = tf.math.square(predict - random)
        mean, stddev = self.moving_average(error)

        modulator = 1 + (error - mean) / stddev
        return modulator
        
