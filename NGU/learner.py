import tensorflow as tf

from models import R2D2


class Learner:
    def __init__(self, num_of_actions):
        self.online = R2D2(num_of_actions)
        self.target = R2D2(num_of_actions)
    
    def get_target_network(self):
        return self.target
    
    def train(self, batch_observations, batch_actions, batch_rewards, batch_discount_factors, batch_next_observations):
        with tf.GradientTape() as tape:
            with tape.stop_recording():
                self.target()
        pass