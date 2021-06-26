import gym
import numpy as np
import tensorflow as tf
from collections import deque
from datetime import datetime

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# env, algo name
env_id = "CartPole-v1"
algo = "ddqn"

# env setup
max_episode = 1500
solved_rewards = 450

# epsilon
epsilon = 1.0
epsilon_fin = 0.01
epsilon_frame = 10000
epsilon_interval = epsilon - epsilon_fin

# replay memory
buffer_size = 10000

# train
batch_size = 64
discount_factor = 0.99
start_step = 5000
target_update_step = 500

# dir name
now = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "runs/{}-{}".format(algo, now)
weights_dir = "checkpoints/{}-{}/{}".format(algo, now, {})
video_dir = "videos/{}-{}".format(algo, now)


def create_model(state_shape, n_actions):
    inputs = tf.keras.Input(shape=state_shape)
    hidden1 = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu)(inputs)
    hidden2 = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu)(hidden1)
    outputs = tf.keras.layers.Dense(units=n_actions)(hidden2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


class ReplayMemory:
    def __init__(self, maxlen=10000):
        self.state = deque([], maxlen)
        self.action = deque([], maxlen)
        self.state_next = deque([], maxlen)
        self.rewards = deque([], maxlen)
        self.done = deque([], maxlen)
    
    def put(self, state, action, state_next, rewards, done):
        self.state.append(state)
        self.action.append(action)
        self.state_next.append(state_next)
        self.rewards.append(rewards)
        self.done.append(done)
    
    def get(self, batch_size=32):
        _state, _action, _state_next, _rewards, _done = [], [], [], [], []
        for i in np.random.choice(len(self.done), size=batch_size):
            _state.append(self.state[i])
            _action.append(self.action[i])
            _state_next.append(self.state_next[i])
            _rewards.append(self.rewards[i])
            _done.append(self.done[i])
        return np.array(_state), np.array(_action), np.array(_state_next), np.array(_rewards), np.array(_done)
    
    def get_length(self):
        return len(self.done)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_mse = tf.keras.losses.MSE


if __name__ == "__main__":
    env = gym.make(env_id)
    env.seed(seed)

    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    model = create_model(state_shape, n_actions)
    model_target = create_model(state_shape, n_actions)
    model_target.set_weights(model.get_weights())
    
    buffer = ReplayMemory(buffer_size)

    # log - tensorboard
    writer = tf.summary.create_file_writer(log_dir)
    sum_rewards = tf.keras.metrics.Sum()
    avg_loss = tf.keras.metrics.Mean()
    avg_q = tf.keras.metrics.Mean()

    # game setup
    episode, step = 0, 0
    rewards_list = deque([], maxlen=100)

    while episode < max_episode:
        done = False
        episode += 1
        rewards_all = 0
        state = env.reset()

        while not done:
            step += 1

            if np.random.sample() < epsilon:
                action = np.random.choice(n_actions)
            else:
                _state = np.expand_dims(state, axis=0)
                action = np.argmax(model(_state).numpy(), axis=1)[0]

            state_next, rewards, done, _ = env.step(action)
            
            buffer.put(state, [action], state_next, [rewards], [done])
            
            sum_rewards(rewards)
            state = state_next

            if step <= start_step:
                continue

            epsilon = max(epsilon_fin, epsilon - epsilon_interval/epsilon_frame)

            # Train DDQN
            state_sample, action_sample, state_next_sample, rewards_sample, done_sample = buffer.get(batch_size)
            done_sample = done_sample.astype(int)

            with tf.GradientTape() as tape:
                action_next = tf.math.argmax(model(state_next_sample), axis=1)
                action_next = tf.expand_dims(action_next, axis=1)
                with tape.stop_recording():
                    q_values_next = model_target(state_next_sample)
                    future_rewards = tf.gather(q_values_next, action_next, batch_dims=1) * (1 - done_sample)
                    q_values_target = rewards_sample + discount_factor * future_rewards
                q_values = model(state_sample)
                q_action = tf.gather(q_values, action_sample, batch_dims=1)
                loss = loss_mse(q_values_target, q_action)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # calc metrics
            avg_loss(loss)
            avg_q(q_action)

            # update target model
            if step % target_update_step == 0:
                model_target.set_weights(model.get_weights())

        rewards_list.append(sum_rewards.result())

        with writer.as_default():
            tf.summary.scalar("Episode/Avg. Q-value", avg_q.result(), step=episode)
            tf.summary.scalar("Episode/Avg. Loss", avg_loss.result(), step=episode)
            tf.summary.scalar("Episode/Rewards", sum_rewards.result(), step=episode)
            tf.summary.scalar("Episode/Epsilon", epsilon, step=episode)
        print("Episode: {} | rewards: {} | buffer: {} | epsilon: {}".format(
            episode, sum_rewards.result(), buffer.get_length(), epsilon))
        
        # reset metrics
        avg_q.reset_states()
        avg_loss.reset_states()
        sum_rewards.reset_states()

        if np.mean(rewards_list) >= solved_rewards:
            model.save_weights(weights_dir.format("solved"))
    model.save_weights(weights_dir.format("done"))


    # Test model & save videos
    from gym.wrappers import Monitor
    
    env = Monitor(env, video_dir, force=True)

    done = False
    state = env.reset()
    while not done:
        # env.render()
        _state = np.expand_dims(state, axis=0)
        action = np.argmax(model(_state).numpy(), axis=1)[0]
        state, _, done, _ = env.step(action)
    env.close()
    