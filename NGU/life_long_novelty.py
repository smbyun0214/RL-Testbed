from moving_average import MovingAverage
import tensorflow as tf

class RND(tf.keras.Model):
    def __init__(self):
        super(RND, self).__init__()
        
        self.cnn1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(
                32, kernel_size=(8, 8), strides=(4, 4),
                activation=tf.keras.activations.relu, padding='same'))
        self.cnn2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(
                64, kernel_size=(4, 4), strides=(2, 2),
                activation=tf.keras.activations.relu, padding='same'))
        self.cnn3 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(
                64, kernel_size=(3, 3), strides=(1, 1),
                activation=tf.keras.activations.relu, padding='same'))
        self.flatten = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Flatten())
        self.dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(128))
    
    def call(self, inputs):
        x = self.cnn1(inputs)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.flatten(x)
        outputs = self.dense(x)
        return outputs


class LifeLongNovelty():
    def __init__(self):
        self.random_network = RND()
        self.predict_network = RND()
        self.moving_average = MovingAverage()
        
        self.random_network.trainable = False

    def reset(self):
        self.moving_average.reset()

    def __call__(self, value):
        random = self.random_network(value)
        predict = self.predict_network(value)

        error = tf.math.square(predict - random)
        mean, stddev = self.moving_average(error)

        modulator = 1 + (error - mean) / stddev
        return modulator