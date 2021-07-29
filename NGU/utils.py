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


def calcuate_beta(i, N, beta=0.3):
    if i == 0:
        return 0.0
    elif i == N - 1:
        return beta

    return beta * tf.nn.sigmoid(
                            10 * (2 * i - (N - 2)) /
                            (N - 2))
    

def calcuate_discount_factor(i, N, min_discount_factor=0.99, max_discount_factor=0.997):
    return 1 - tf.math.exp(
                        (N - 1 - i) * tf.math.log(1 - max_discount_factor) + i * tf.math.log(1 - min_discount_factor) /
                        (N - 1))