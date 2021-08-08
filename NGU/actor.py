import tensorflow as tf

from collections import deque
from models import R2D2
from novelty import Episodic, LifeLong
from utils import calcuate_beta, calcuate_discount_factor
from ext_typing import Transition
from typing import List


class Actor:
    def __init__(self, num_of_actions, behavior_policy, life_long: LifeLong, index, N=32, maximum_reward_scaling=5):
        self.maximum_reward_scaling = maximum_reward_scaling

        self.beta = calcuate_beta(index % N, N)
        self.discount_factor = calcuate_discount_factor(index % N, N)

        self.beta = tf.expand_dims([ self.beta ], axis=0)
        self.discount_factor = tf.expand_dims([ self.discount_factor ], axis=0)

        self.behavior_policy = behavior_policy
        self.life_long = life_long
        self.episodic = Episodic(num_of_actions)
        self.optimizer = tf.keras.optimizers.Adam()

    def reset(self):
        self.episodic.reset()

    def get_action(self, observations, prev_actions, prev_episodic_reward, prev_intrinsic_reward):
        action, hidden_state = self.behavior_policy(observations, prev_actions, prev_episodic_reward, prev_intrinsic_reward, self.beta)
        return action, hidden_state
    
    def get_intrinsic_reward(self, observations, next_observations):
        episodic_reward = self.episodic.get_similarity(observations, next_observations)
        modulator = self.life_long.get_modulator(observations)
        intrinsic_reward = episodic_reward * tf.math.minimum(tf.math.maximum(modulator, 1), self.maximum_reward_scaling)
        return intrinsic_reward

    def train(self, batch_transitions: List[Transition]):
        self.episodic.train(batch_transitions)
        self.life_long.train(batch_transitions)
