import tensorflow as tf


class Embedding(tf.keras.Model):
    def __init__(self, num_of_actions):
        super(Embedding, self).__init__()

        self.common1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, kernel_size=(8, 8), strides=(4, 4),
                activation=tf.keras.activations.relu, padding='same'),
            tf.keras.layers.Conv2D(
                64, kernel_size=(4, 4), strides=(2, 2),
                activation=tf.keras.activations.relu, padding='same'),
            tf.keras.layers.Conv2D(
                64, kernel_size=(3, 3), strides=(1, 1),
                activation=tf.keras.activations.relu, padding='same'),
            tf.keras.layers.Dense(32, activation=tf.keras.activations.relu)
        ])
        self.common2 = tf.keras.models.clone_model(self.common1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(num_of_actions, activation=tf.keras.activations.softmax)
    
    def call(self, observations, next_observations):
        x1 = self.common1(observations)
        x2 = self.common2(next_observations)

        flatten_x1 = self.flatten(x1)
        flatten_x2 = self.flatten(x2)

        controllable_states = tf.concat([flatten_x1, flatten_x2], axis=1)

        x = self.dense1(controllable_states)
        outputs = self.dense2(x)
        return outputs, controllable_states


class RND(tf.keras.Model):
    def __init__(self):
        super(RND, self).__init__()
        
        self.common = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, kernel_size=(8, 8), strides=(4, 4),
                activation=tf.keras.activations.relu, padding='same'),
            tf.keras.layers.Conv2D(
                64, kernel_size=(4, 4), strides=(2, 2),
                activation=tf.keras.activations.relu, padding='same'),
            tf.keras.layers.Conv2D(
                64, kernel_size=(3, 3), strides=(1, 1),
                activation=tf.keras.activations.relu, padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128)])
    
    def call(self, observations):
        outputs = self.common(observations)
        return outputs


class R2D2(tf.keras.Model):
    def __init__(self, num_of_actions):
        super(R2D2, self).__init__()
        self.num_of_actions = num_of_actions

        self.common = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, kernel_size=(8, 8), strides=(4, 4),
                activation=tf.keras.activations.relu, padding='same'),
            tf.keras.layers.Conv2D(
                64, kernel_size=(4, 4), strides=(2, 2),
                activation=tf.keras.activations.relu, padding='same'),
            tf.keras.layers.Conv2D(
                64, kernel_size=(3, 3), strides=(1, 1),
                activation=tf.keras.activations.relu, padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                512, activation=tf.keras.activations.relu)])
        
        self.lstm = tf.keras.layers.LSTM(512, return_state=True)

        self.value1 = tf.keras.Sequential([
            tf.keras.layers.Dense(
                512, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(1)])
        
        self.value2 = tf.keras.Sequential([
            tf.keras.layers.Dense(
                512, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(num_of_actions)])

    def reset(self):
        self.lstm.reset_states()
    
    def call(self, observations, prev_actions, prev_episodic_reward, prev_intrinsic_reward, beta):
        x = self.common(observations)

        one_hot_actions = tf.one_hot(prev_actions, self.num_of_actions)
        squeezed_actions = tf.squeeze(one_hot_actions, axis=1)
        print(x.shape, squeezed_actions.shape, prev_episodic_reward.shape, prev_intrinsic_reward.shape, beta.shape)
        print(x.dtype, squeezed_actions.dtype, prev_episodic_reward.dtype, prev_intrinsic_reward.dtype, beta.dtype)
        concat = tf.concat([x, squeezed_actions, prev_episodic_reward, prev_intrinsic_reward, beta], axis=1)
        sequence = tf.expand_dims(concat, axis=1)
        
        x, memory_state, carry_state = self.lstm(sequence)
        print(x.shape)
        hidden_state = (memory_state, carry_state)

        value1 = self.value1(x)
        value2 = self.value2(x)
        mean = tf.reduce_mean(value2, axis=1)

        outputs = value1 + value2 - mean
        return outputs, hidden_state