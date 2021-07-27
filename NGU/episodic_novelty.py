from moving_average import MovingAverage
import tensorflow as tf

class Embedding(tf.keras.Model):
    def __init__(self, num_of_actions):
        super(Embedding, self).__init__()

        self.common1 = tf.keras.Sequential([
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    32, kernel_size=(8, 8), strides=(4, 4),
                    activation=tf.keras.activations.relu, padding='same')),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    64, kernel_size=(4, 4), strides=(2, 2),
                    activation=tf.keras.activations.relu, padding='same')),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    64, kernel_size=(3, 3), strides=(1, 1),
                    activation=tf.keras.activations.relu, padding='same')),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(32, activation=tf.keras.activations.relu))
        ])
        self.common2 = tf.keras.models.clone_model(self.common1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(num_of_actions, activation=tf.keras.activations.softmax)
    
    def call(self, obs, next_obs):
        x1 = self.common1(obs)
        x2 = self.common2(next_obs)
        x = tf.concat([x1, x2], axis=2)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class EpisodicNovelty():
    def __init__(self, num_of_actions, c=0.001, epsilon=0.001):
        self.c = c
        self.epsilon = epsilon

        self.buffer = []
        self.network = Embedding(num_of_actions)
        self.moving_average = MovingAverage()


    def reset(self):
        self.buffer = []
    

    def __call__(self, obs, next_obs):
        self.moving_average.reset()

        controllable_state = self.network(obs, next_obs)

        denominator = tf.constant(0, dtype=tf.float32)

        for state in self.buffer:
            d2 = tf.norm(controllable_state - state)
            mean, _ = self.moving_average(d2)
            inverse_kernel = self.epsilon / ((d2 / mean) + self.epsilon)
            denominator += inverse_kernel
        
        intrinsic = 1 / (tf.sqrt(denominator) + self.c)

        self.buffer.append(controllable_state)

        return 1 / intrinsic