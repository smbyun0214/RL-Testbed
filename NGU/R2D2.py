import tensorflow as tf

class R2D2(tf.keras.Model):
    def __init__(self, num_of_actions):
        super(R2D2, self).__init__()
        self.num_of_actions = num_of_actions

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
            tf.keras.layers.Dense(
                512, activation=tf.keras.activations.relu))
        
        self.lstm = tf.keras.layers.LSTM(512)

        self.dense11 = tf.keras.layers.Dense(
            512, activation=tf.keras.activations.relu)
        self.dense12 = tf.keras.layers.Dense(1)
        
        self.dense21 = tf.keras.layers.Dense(
            512, activation=tf.keras.activations.relu)
        self.dense22 = tf.keras.layers.Dense(18)
    
    def call(self, x, a, re, ri, beta):
        input = x
        x = self.cnn1(input)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.flatten(x)
        x = self.dense(x)

        a = tf.one_hot(a, self.num_of_actions)
        print(x.shape, a.shape, re.shape, ri.shape, beta.shape)
        concat = tf.concat([x, a, re, ri, beta], axis=2)
        
        x = self.lstm(concat)

        x1 = self.dense11(x)
        x1 = self.dense12(x1)

        x2 = self.dense21(x)
        x2 = self.dense22(x2)

        output = x1 + x2 - tf.math.reduce_mean(x2, axis=1)
        return output