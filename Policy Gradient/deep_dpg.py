import gym
import tqdm
import numpy as np
import tensorflow as tf

from datetime import datetime
from collections import deque
from typing import List, Tuple


class ReplayBuffer():
    def __init__(self, buffer_size) -> None:
        self.observations0 = deque([], maxlen=buffer_size)
        self.observations1 = deque([], maxlen=buffer_size)
        self.actions = deque([], maxlen=buffer_size)
        self.rewards = deque([], maxlen=buffer_size)
        self.huber_loss = tf.keras.losses.Huber()
    
    def reset(self):
        self.observations0.clear()
        self.observations1.clear()
        self.actions.clear()
        self.rewards.clear()
    
    def remember(self, observation0: np.ndarray, action: np.ndarray, reward: np.ndarray, observation1: np.ndarray):
        self.observations0.append(observation0)
        self.observations1.append(observation1)
        self.actions.append(action)
        self.rewards.append(reward)

    def get_tensor(self, batch_indices) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        observations0 = [ self.observations0[idx] for idx in batch_indices ]
        observations1 = [ self.observations1[idx] for idx in batch_indices ]
        actions = [ self.actions[idx] for idx in batch_indices ]
        rewards = [ self.rewards[idx] for idx in batch_indices ]

        observations0 = tf.convert_to_tensor(observations0, dtype=tf.float32)
        observations1 = tf.convert_to_tensor(observations1, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        rewards = tf.expand_dims(rewards, axis=1)
        
        return (observations0, observations1, actions, rewards)

    def learn(
        self,
        actor_model: tf.keras.Model,
        critic_model: tf.keras.Model,
        target_actor_model: tf.keras.Model,
        target_critic_model: tf.keras.Model,
        actor_optimizer: tf.keras.optimizers.Optimizer,
        critic_optimizer: tf.keras.optimizers.Optimizer,
        batch_size: int = 64,
        discount_factor: float = 0.99,
        tau: float = 0.005):
        batch_indices = np.random.choice(len(self.rewards), batch_size)
        observations0, observations1, actions, rewards = self.get_tensor(batch_indices)

        self.training(
            actor_model, critic_model,
            target_actor_model, target_critic_model,
            actor_optimizer,
            critic_optimizer,
            observations0, observations1, actions, rewards,
            discount_factor, tau)

    @tf.function
    def training(
        self,
        actor_model: tf.keras.Model,
        critic_model: tf.keras.Model,
        target_actor_model: tf.keras.Model,
        target_critic_model: tf.keras.Model,
        actor_optimizer: tf.keras.optimizers.Optimizer,
        critic_optimizer: tf.keras.optimizers.Optimizer,
        observations0: tf.Tensor,
        observations1: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        discount_factor: float,
        tau: float):
        with tf.GradientTape() as tape:
            target_actions = target_actor_model(observations1)
            y = rewards + discount_factor * target_critic_model([observations1, target_actions])
            values = critic_model([observations0, actions])
            critic_loss = self.huber_loss(y, values)

        critic_grads = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grads, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(observations0)
            values = critic_model([observations0, actions])
            actor_loss = -tf.math.reduce_mean(values)
        
        actor_grads = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grads, actor_model.trainable_variables))

        self.update_target(target_actor_model, actor_model, tau)
        self.update_target(target_critic_model, critic_model, tau)

    @tf.function
    def update_target(
        self,
        target_model: tf.keras.Model,
        model: tf.keras.Model,
        tau: float):
        for (target_weights, weights) in zip(target_model.variables, model.variables):
            target_weights.assign(tau * weights + (1 - tau) * target_weights)


def get_actor_model(state_shape: Tuple, output_units: Tuple, lower_bound: float, upper_bound: float) -> tf.keras.Model:
    inputs = tf.keras.Input(state_shape)
    hidden = tf.keras.layers.Dense(128, activation="relu")(inputs)
    hidden = tf.keras.layers.Dense(128, activation="relu")(hidden)
    outputs = tf.keras.layers.Dense(output_units, kernel_initializer="zeros")(hidden)
    outputs = tf.clip_by_value(outputs, lower_bound, upper_bound)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

def get_critic_model(state_shape: Tuple, action_shape: Tuple):
    state_inputs = tf.keras.Input(state_shape)
    hidden0 = tf.keras.layers.Dense(128, activation="relu")(state_inputs)

    action_inputs = tf.keras.Input(action_shape)
    hidden1 = tf.keras.layers.Dense(128, activation="relu")(action_inputs)

    merged = tf.keras.layers.Concatenate()([hidden0, hidden1])

    hidden2 = tf.keras.layers.Dense(128, activation="relu")(merged)
    hidden2 = tf.keras.layers.Dense(128, activation="relu")(hidden2)
    values = tf.keras.layers.Dense(1, kernel_initializer="zeros")(hidden2)

    return tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=values)

class OUActionNoise:
    def __init__(
        self,
        mean: tf.Tensor,
        stddev: tf.Tensor,
        theta: float = 0.15,
        dt: float = 1e-2,
        x0=None):
        self.mean = mean
        self.stddev = stddev
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = tf.zeros_like(self.mean)
    
    @tf.function
    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.stddev * tf.math.sqrt(self.dt) * tf.random.normal(self.mean.shape))
        self.x_prev = x
        return x

@tf.function
def get_action(state: tf.Tensor, actor_model: tf.keras.Model, noise: OUActionNoise) -> Tuple[tf.Tensor]:
    return actor_model(state) + noise()

def env_step(env: gym.Env, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, reward, done, _ = env.step(actions)
    return state.astype(np.float32), np.array(reward, dtype=np.float32), np.array(done, dtype=bool)


def run_episode(
    initial_state: np.ndarray,
    env: gym.Env,
    actor_model: tf.keras.Model,
    critic_model: tf.keras.Model,
    target_actor_model: tf.keras.Model,
    target_critic_model: tf.keras.Model,
    noise: OUActionNoise,
    actor_optimizer: tf.keras.optimizers.Optimizer,
    critic_optimizer: tf.keras.optimizers.Optimizer,
    buffer: ReplayBuffer,
    max_step: int) -> int:
    initial_state_shape = initial_state.shape
    state = initial_state

    episode_rewards = 0

    for t in range(max_step):
        state = np.expand_dims(state, axis=0)
        tf_state = tf.convert_to_tensor(state, dtype=tf.float32)
        tf_actions = get_action(tf_state, actor_model, noise)
        
        actions = tf_actions.numpy()
        next_state, reward, done = env_step(env, actions[0])
        
        state = state.reshape(initial_state_shape)
        next_state = next_state.reshape(initial_state_shape)
        buffer.remember(state, actions[0], reward, next_state)

        buffer.learn(actor_model, critic_model, target_actor_model, target_critic_model, actor_optimizer, critic_optimizer)

        episode_rewards += reward
        state = next_state

        if done:
            break

    return episode_rewards


def main():
    algorithm = "Deep_DPG"
    env_id = "Pendulum-v0"
    buffer_size = 50000
    max_episodes = 50000
    max_step = 200
    seed = 42
    min_episodes_criterion = 100
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    today = datetime.today().strftime("%Y-%m-%d-%H%M%S")

    np.random.seed(seed)

    env = gym.make(env_id)
    env.seed(seed)
    lower_bound = env.action_space.low.item()
    upper_bound = env.action_space.high.item()
    observation_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]

    tf.random.set_seed(seed)
    
    actor_model = get_actor_model(observation_shape, action_shape, lower_bound, upper_bound)
    critic_model = get_critic_model(observation_shape, action_shape)
    
    target_actor_model = get_actor_model(observation_shape, action_shape, lower_bound, upper_bound)
    target_critic_model = get_critic_model(observation_shape, action_shape)

    actor_model.set_weights(target_actor_model.get_weights())
    critic_model.set_weights(target_critic_model.get_weights())

    buffer = ReplayBuffer(buffer_size)
    noise = OUActionNoise(mean=tf.zeros(action_shape), stddev=tf.ones(action_shape)*0.2)

    episodes_reward = deque(maxlen=min_episodes_criterion)
    writer = tf.summary.create_file_writer(f"/content/drive/MyDrive/Train/runs/{algorithm}-{today}")

    with tqdm.trange(max_episodes) as t:
        for i in t:
            initial_state = env.reset()
            reward = run_episode(initial_state, env, actor_model, critic_model, target_actor_model, target_critic_model, noise, actor_optimizer, critic_optimizer, buffer, max_step)
            episodes_reward.append(reward)
            running_reward = np.mean(episodes_reward)

            t.set_description(f"Episode {i}")
            t.set_postfix_str(f" | episode_reward: {reward:.2f}, running_reward: {running_reward:.2f}")
        
            with writer.as_default(step=i):
                tf.summary.scalar(f"Avg. last {min_episodes_criterion} episode reward", running_reward)
    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


if __name__ == "__main__":
    main()