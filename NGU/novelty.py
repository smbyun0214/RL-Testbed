import tensorflow as tf

from collections import deque
from buffer import ActorBuffer
from utils import MovingAverage
from models import Embedding, RND
from ext_typing import Observation, Action, Transition
from typing import Tuple, List

class Episodic():
    def __init__(
        self, num_of_actions, epsilon=0.0001, num_of_neighbours=10, cluster_distance=0.008,
        pseudo_counts=0.001, maximum_similarity=8, episodic_memory_capacity=30000):
        self.epsilon = epsilon
        self.num_of_neighbours = num_of_neighbours
        self.cluster_distance = cluster_distance
        self.pseudo_counts = pseudo_counts
        self.maximum_similarity = maximum_similarity

        self.episodic_memory = deque([], maxlen=episodic_memory_capacity)
        self.moving_average = MovingAverage()
        self.network = Embedding(num_of_actions)
        self.optimizer = tf.keras.optimizers.Adam()


    def reset(self):
        self.episodic_memory.clear()
        self.moving_average.reset()


    def get_similarity(self, observations: Observation, next_observations: Observation) -> Tuple[List[tf.float32], float]:
        likelihood, controllable_state = self.network(observations, next_observations)

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
            return likelihood, 0.0
        else:
            return likelihood, 1 / similarity
        
    def train(self, batch_transitions: List[Transition]):
        observations = []
        next_observations = []
        actions = []
        for transition in batch_transitions:
            _observations = transition.observations[-6]
            _actions = transition.actions[-5]
            
            for _observation, _next_observation, _action in zip(_observations[:-1], _observations[1:], _actions):
                observations.append(_observation)
                next_observations.append(_next_observation)
                actions.append(_action)
                
        observations = tf.convert_to_tensor(observations)
        next_observations = tf.convert_to_tensor(next_observations)
        actions = tf.convert_to_tensor(actions)
        with tf.GradientTape() as tape:
            likelihood, _ = self.network(observations, next_observations)
            loss = tf.keras.losses.mean_squared_error(actions, likelihood)
        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        

class LifeLong():
    def __init__(self):
        self.random_network = RND()
        self.predict_network = RND()
        self.moving_average = MovingAverage()
        
        self.random_network.trainable = False

    def reset(self):
        self.moving_average.reset()

    def get_modulator(self, observations):
        random = self.random_network(observations)
        predict = self.predict_network(observations)

        error = tf.math.square(predict - random)
        mean, stddev = self.moving_average(error)

        modulator = 1 + (error - mean) / stddev
        return modulator
    
    def train(self, batch_transitions: List[Transition]):
        observations = []
        next_observations = []

        for transition in batch_transitions:
            _observations = transition.observations[-6]
            
            for _observation, _next_observation in zip(_observations[:-1], _observations[1:]):
                observations.append(_observation)
                next_observations.append(_next_observation)
                
        observations = tf.convert_to_tensor(observations)
        next_observations = tf.convert_to_tensor(next_observations)
        with tf.GradientTape() as tape:
            with tape.stop_recording():
                true = self.random_network(observations)
            predict = self.predict_network(observations)
            loss = tf.keras.losses.mean_squared_error(true, predict)
        grads = tape.gradient(loss, self.predict_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.predict_network.trainable_variables))
