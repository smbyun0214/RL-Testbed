import tensorflow as tf

from models import R2D2
from buffer import LearnerBuffer

class Learner:
    def __init__(self, num_of_actions, buffer: LearnerBuffer):
        self.online = R2D2(num_of_actions)
        self.target = R2D2(num_of_actions)
        self.buffer = buffer
        
    def get_target_network(self):
        return self.target
    
    def train(self):
        transitions = self.buffer.get_samples()
        
        with tf.GradientTape() as tape:
            with tape.stop_recording():
                self.target()
        pass