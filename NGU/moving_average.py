import numpy as np
import tensorflow as tf

class MovingAverage():
    def __init__(self):
        self.eps = tf.constant(np.finfo(np.float32).eps)
        self.count = 0
        self.mean = None
        self.M2 = None

    def reset(self):
        self.count = 0
        self.mean = None
        self.M2 = None

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    def __call__(self, value):
        if self.count == 0:
            self.mean = tf.zeros_like(value, dtype=tf.float32)
            self.M2 = tf.zeros_like(value, dtype=tf.float32)
        
        self.count += 1
        
        delta0 = value - self.mean
        self.mean += delta0 / self.count

        delta1 = value - self.mean
        self.M2 += delta0 * delta1

        if self.count > 1:
            return self.mean, tf.math.sqrt(self.M2/self.count)
        
        return self.mean + self.eps, tf.zeros_like(value, dtype=tf.float32) + self.eps