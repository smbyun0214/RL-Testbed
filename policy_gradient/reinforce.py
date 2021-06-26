import gym
import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from datetime import datetime
from collections import deque
from typing import List, Tuple


class EpisodicBuffer():
    def __init__(self) -> None:
        self.observations0 = []
        self.observations1 = []
        self.actions = []
        self.rewards = []
    
    def reset(self):
        self.observations0 = []
        self.observations1 = []
        self.actions = []
        self.rewards = []
    
    def remember(self, observation0: np.ndarray, action: np.ndarray, reward: np.ndarray, observation1: np.ndarray):
        self.observations0.append(observation0)
        self.observations1.append(observation1)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def get_expected_return(self, discount_factor: float) -> List[np.ndarray]:
        returns = []

        rewards = self.rewards[::-1]
        discounted_sum = 0
        discounted_sum_shape = rewards[0].shape
        for reward in rewards:
            discounted_sum = reward + discount_factor * discounted_sum
            discounted_sum = discounted_sum.reshape(discounted_sum_shape)
            returns.append(discounted_sum)
        
        return returns[::-1]

    def get_tensor(self, discount_factor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        observations0 = tf.convert_to_tensor(self.observations0, dtype=tf.float32)
        observations1 = tf.convert_to_tensor(self.observations1, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.actions, dtype=tf.float32)

        returns = self.get_expected_return(discount_factor)
        returns = tf.convert_to_tensor(np.expand_dims(returns, 1), dtype=tf.float32)
        
        return (observations0, observations1, actions, returns)

    def learn(self, policy_model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, discount_factor=0.99):
        observations0, _, actions, returns = self.get_tensor(discount_factor)
        self.training(policy_model, optimizer, observations0, actions, returns)

    @tf.function
    def training(self, policy_model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, observations0: tf.Tensor, actions: tf.Tensor, returns:tf.Tensor):
        with tf.GradientTape() as tape:
            means, log_stds = policy_model(observations0)
            stddevs = tf.math.exp(log_stds)
            
            dists = tfp.distributions.Normal(means, stddevs)
            log_probs = dists.log_prob(actions)

            policy_loss = -tf.math.reduce_mean(log_probs * returns)
        
        policy_grads = tape.gradient(policy_loss, policy_model.trainable_variables)
        optimizer.apply_gradients(zip(policy_grads, policy_model.trainable_variables))


def get_policy_model(input_shape: Tuple, output_units: Tuple, lower_bound: float, upper_bound: float) -> tf.keras.Model:
    inputs = tf.keras.Input(input_shape)
    hidden = tf.keras.layers.Dense(128, activation="relu")(inputs)
    hidden = tf.keras.layers.Dense(128, activation="relu")(hidden)

    means = tf.keras.layers.Dense(output_units, kernel_initializer="zeros")(hidden)
    log_stds = tf.keras.layers.Dense(output_units, kernel_initializer="zeros")(hidden)

    means = tf.clip_by_value(means, lower_bound, upper_bound)
    log_stds = tf.clip_by_value(log_stds, -tf.math.log(upper_bound), tf.math.log(upper_bound))

    return tf.keras.Model(inputs=inputs, outputs=[means, log_stds])

@tf.function
def get_action(state: tf.Tensor, policy_model: tf.keras.Model, training: bool = True) -> Tuple[tf.Tensor]:
    means, log_stds = policy_model(state)
    stddevs = tf.math.exp(log_stds)

    if training:
        actions = tf.random.normal(tf.shape(means), means, stddevs)
    else:
        actions = means

    return actions

def env_step(env: gym.Env, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, reward, done, _ = env.step(actions)
    return state.astype(np.float32), np.array(reward, dtype=np.float32), np.array(done, dtype=bool)


def run_episode(initial_state: np.ndarray, env: gym.Env, policy_model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, buffer: EpisodicBuffer, max_step: int) -> int:
    initial_state_shape = initial_state.shape
    state = initial_state

    episode_rewards = 0

    for t in range(max_step):
        state = np.expand_dims(state, axis=0)
        tf_state = tf.convert_to_tensor(state, dtype=tf.float32)
        tf_actions = get_action(tf_state, policy_model)
        
        actions = tf_actions.numpy()
        next_state, reward, done = env_step(env, actions[0])
        
        state = state.reshape(initial_state_shape)
        next_state = next_state.reshape(initial_state_shape)
        buffer.remember(state, actions[0], reward, next_state)

        episode_rewards += reward
        state = next_state

        if done:
            break

    buffer.learn(policy_model, optimizer)

    return episode_rewards


def main():
    algorithm = "REINFORCE"
    env_id = "Pendulum-v0"

    max_episodes = 50000
    max_step = 200
    seed = 42
    min_episodes_criterion = 100
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    today = datetime.today().strftime("%Y-%m-%d-%H%M%S")
    best_reward = np.iinfo(np.int32).min

    np.random.seed(seed)

    env = gym.make(env_id)
    env.seed(seed)
    lower_bound = env.action_space.low.item()
    upper_bound = env.action_space.high.item()

    tf.random.set_seed(seed)
    policy_model = get_policy_model(env.observation_space.shape, env.action_space.shape[0], lower_bound, upper_bound)
    buffer = EpisodicBuffer()

    episodes_reward = deque(maxlen=min_episodes_criterion)
    writer = tf.summary.create_file_writer(f"runs/{algorithm}-{today}")

    with tqdm.trange(max_episodes) as t:
        for i in t:
            buffer.reset()
            initial_state = env.reset()
            reward = run_episode(initial_state, env, policy_model, optimizer, buffer, max_step)
            episodes_reward.append(reward)
            running_reward = np.mean(episodes_reward)

            t.set_description(f"Episode {i}")
            t.set_postfix_str(f" | episode_reward: {reward:.2f}, running_reward: {running_reward:.2f}")
        
            with writer.as_default(step=i):
                tf.summary.scalar(f"Avg. last {min_episodes_criterion} episode reward", running_reward)

            if best_reward < int(running_reward):
                best_reward = int(running_reward)
                sign = "m" if np.sign(best_reward) < 0 else ""
                policy_model.save_weights(f"checkpoints/{algorithm}-{env_id}/reward-{sign}{abs(best_reward)}")

    policy_model.save_weights(f"checkpoints/{algorithm}-{env_id}/latest")
    env.close()

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


def save_video():
    from gym.wrappers import Monitor

    algorithm = "REINFORCE"
    env_id = "Pendulum-v0"

    env = gym.make(env_id)
    env = Monitor(env, f"videos/{algorithm}-{env_id}", force=True)

    lower_bound = env.action_space.low.item()
    upper_bound = env.action_space.high.item()

    model = get_policy_model(env.observation_space.shape, env.action_space.shape[0], lower_bound, upper_bound)
    model.load_weights(f"checkpoints/{algorithm}-{env_id}/latest")

    state = env.reset()
    while True:
        tf_state = tf.expand_dims(state, axis=0)
        action = get_action(tf_state, model, training=False)
        state, _, done = env_step(env, action[0])

        if done:
            break
    env.close()
        


if __name__ == "__main__":
    # main()
    save_video()