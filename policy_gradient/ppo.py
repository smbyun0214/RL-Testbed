import gym
import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from datetime import datetime
from collections import deque
from typing import List, Tuple

eps = np.finfo(np.float32).eps


class Actor(tf.keras.Model):
    def __init__(self, output_units: Tuple, lower_bound: float, upper_bound: float):
        super(Actor, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.hidden0 = tf.keras.layers.Dense(128, activation="relu")
        self.hidden1 = tf.keras.layers.Dense(128, activation="relu")
        self.means = tf.keras.layers.Dense(output_units, kernel_initializer="zeros")
        self.log_stddevs = tf.Variable(tf.zeros(output_units))

    def call(self, inputs: tf.Tensor):
        hidden = self.hidden0(inputs)
        hidden = self.hidden1(hidden)
        means = self.means(hidden)
        means = tf.clip_by_value(means, self.lower_bound, self.upper_bound)
        return means, self.log_stddevs


def get_critic_model(state_shape: Tuple):
    inputs = tf.keras.Input(state_shape)
    hidden = tf.keras.layers.Dense(128, activation="relu")(inputs)
    hidden = tf.keras.layers.Dense(128, activation="relu")(hidden)
    values = tf.keras.layers.Dense(1, kernel_initializer="zeros")(hidden)

    return tf.keras.Model(inputs=inputs, outputs=values)


@tf.function
def get_action(state: tf.Tensor, actor_model: tf.keras.Model, training: bool = True) -> Tuple[tf.Tensor]:
    means, log_stddevs = actor_model(state)
    stddevs = tf.math.exp(log_stddevs)

    if training:
        actions = tf.random.normal(tf.shape(means), means, stddevs)
    else:
        actions = means

    return actions


def get_dists(
    actor_model: tf.keras.Model,
    states: tf.Tensor) -> tfp.distributions.Normal:
    means, log_stddevs = actor_model(states)
    stddevs = tf.math.exp(log_stddevs)
    dists = tfp.distributions.Normal(means, stddevs)
    return dists


class EpisodicBuffer():
    def __init__(self) -> None:
        self.observations0 = []
        self.observations1 = []
        self.actions = []
        self.rewards = []

        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.tf_observations0 = None
        self.tf_observations1 = None
        self.tf_actions = None
        self.tf_returns = None
        self.tf_advantages = None
    
    def reset(self):
        self.observations0 = []
        self.observations1 = []
        self.actions = []
        self.rewards = []

    def reset_tf(self):
        self.tf_observations0 = None
        self.tf_observations1 = None
        self.tf_actions = None
        self.tf_returns = None
        self.tf_advantages = None
    
    def remember(self, observation0: np.ndarray, action: np.ndarray, reward: np.ndarray, observation1: np.ndarray):
        self.observations0.append(observation0)
        self.observations1.append(observation1)
        self.actions.append(action)
        self.rewards.append(reward)

    def store_tensor(self, observations0, observations1, actions, returns, advantages):
        self.tf_observations0 = observations0 if self.tf_observations0 is None else tf.concat([self.tf_observations0, observations0], axis=0)
        self.tf_observations1 = observations1 if self.tf_observations1 is None else tf.concat([self.tf_observations1, observations1], axis=0)
        self.tf_actions = actions if self.tf_actions is None else tf.concat([self.tf_actions, actions], axis=0)
        self.tf_returns = returns if self.tf_returns is None else tf.concat([self.tf_returns, returns], axis=0)
        self.tf_advantages = advantages if self.tf_advantages is None else tf.concat([self.tf_advantages, advantages], axis=0)
    
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

    def get_generalized_advantage_estimation(self, critic_model: tf.keras.Model, discount_factor: float, lambda_: float = 0.95) -> List[np.ndarray]:
        gaes = []
        tf_observations0 = tf.convert_to_tensor(self.observations0[::-1])
        tf_observations1 = tf.convert_to_tensor(self.observations1[::-1])
        values0 = critic_model(tf_observations0).numpy()
        values1 = critic_model(tf_observations1).numpy()
        rewards = self.rewards[::-1]

        discounted_sum = 0
        discounted_sum_shape = rewards[0].shape
        for reward, value0, value1 in zip(rewards, values0, values1):
            delta = reward + discount_factor*value1 - value0
            discounted_sum = delta + discount_factor*lambda_*discounted_sum
            discounted_sum = discounted_sum.reshape(discounted_sum_shape)
            gaes.append(discounted_sum)

        return gaes[::-1]


    def get_tensor(self, critic_model: tf.keras.Model, discount_factor=0.99) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        observations0 = tf.convert_to_tensor(self.observations0, dtype=tf.float32)
        observations1 = tf.convert_to_tensor(self.observations1, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.actions, dtype=tf.float32)

        returns = self.get_expected_return(discount_factor)
        returns = tf.convert_to_tensor(np.expand_dims(returns, 1), dtype=tf.float32)
        
        advantages = self.get_generalized_advantage_estimation(critic_model, discount_factor)
        advantages = tf.convert_to_tensor(np.expand_dims(advantages, 1), dtype=tf.float32)
        advantages = (advantages - tf.math.reduce_mean(advantages)) / tf.math.reduce_std(advantages)
        
        return (observations0, observations1, actions, returns, advantages)

    def learn(
        self,
        actor_model: tf.keras.Model,
        critic_model: tf.keras.Model,
        actor_optimizer: tf.keras.optimizers.Optimizer,
        critic_optimizer: tf.keras.optimizers.Optimizer,
        epsilon: float = 0.2,
        vf_coeff: float = 1.0,
        entropy_coeff: float = 0.01,
        batch_size: int = 64,
        iter: int = 5):
        observations0, actions, returns, advantages = self.tf_observations0, self.tf_actions, self.tf_returns, self.tf_advantages
        self.training(actor_model, critic_model, actor_optimizer, critic_optimizer, observations0, actions, returns, advantages, epsilon, vf_coeff, entropy_coeff, batch_size, iter)

    @tf.function
    def training(
        self,
        actor_model: tf.keras.Model,
        critic_model: tf.keras.Model,
        actor_optimizer: tf.keras.optimizers.Optimizer,
        critic_optimizer: tf.keras.optimizers.Optimizer,
        observations0: tf.Tensor,
        actions: tf.Tensor,
        returns: tf.Tensor,
        advantages: tf.Tensor,
        epsilon: float,
        vf_coeff: float,
        entropy_coeff: float,
        batch_size: int,
        iter: int):

        length = returns.shape[0]

        old_dists = get_dists(actor_model, observations0)
        old_log_probs = old_dists.log_prob(actions)
        
        for _ in tf.range(iter):
            batch_indices = tf.random.categorical(tf.zeros((1, length)), batch_size)[0]

            batch_observations0 = tf.gather(observations0, batch_indices)
            batch_actions = tf.gather(actions, batch_indices)
            batch_returns = tf.gather(returns, batch_indices)
            batch_advantages = tf.gather(advantages, batch_indices)
            batch_log_probs = tf.gather(old_log_probs, batch_indices)

            with tf.GradientTape(persistent=True) as tape:
                new_dists = get_dists(actor_model, batch_observations0)
                new_log_probs = new_dists.log_prob(batch_actions)
                batch_values = critic_model(batch_observations0)

                ratio = tf.math.exp(new_log_probs - batch_log_probs)
                surrogate1 = ratio * batch_advantages
                surrogate2 = tf.clip_by_value(ratio, 1.0 - epsilon, 1.0 + epsilon) * batch_advantages
                surrogate = -tf.math.minimum(surrogate1, surrogate2)

                vf_loss = vf_coeff * tf.math.square(batch_values - batch_returns)

                entropy = -entropy_coeff * new_dists.entropy()

                total_loss = surrogate + vf_loss + entropy

                # tf.print("surrogate:", surrogate.shape, "vf_loss:", vf_loss.shape, "entropy:", entropy.shape)

            actor_grads = tape.gradient(total_loss, actor_model.trainable_variables)
            critic_grads = tape.gradient(total_loss, critic_model.trainable_variables)
            
            actor_optimizer.apply_gradients(zip(actor_grads, actor_model.trainable_variables))
            critic_optimizer.apply_gradients(zip(critic_grads, critic_model.trainable_variables))


def env_step(env: gym.Env, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, reward, done, _ = env.step(actions)
    return state.astype(np.float32), np.array(reward, dtype=np.float32), np.array(done, dtype=bool)


def run_episode(
    initial_state: np.ndarray,
    env: gym.Env,
    actor_model: tf.keras.Model,
    buffer: EpisodicBuffer,
    max_step: int) -> int:
    initial_state_shape = initial_state.shape
    state = initial_state

    episode_rewards = 0

    for t in range(max_step):
        state = np.expand_dims(state, axis=0)
        tf_state = tf.convert_to_tensor(state, dtype=tf.float32)
        tf_actions = get_action(tf_state, actor_model)
        
        actions = tf_actions.numpy()
        next_state, reward, done = env_step(env, actions[0])
        
        state = state.reshape(initial_state_shape)
        next_state = next_state.reshape(initial_state_shape)
        buffer.remember(state, actions[0], reward, next_state)

        episode_rewards += reward
        state = next_state

        if done:
            break

    return episode_rewards



def main():
    algorithm = "PPO"
    env_id = "Pendulum-v0"

    max_episodes = 50000
    max_step = 200
    seed = 42
    min_episodes_criterion = 100
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    today = datetime.today().strftime("%Y-%m-%d-%H%M%S")
    best_reward = np.iinfo(np.int32).min

    np.random.seed(seed)

    env = gym.make(env_id)
    env.seed(seed)
    lower_bound = env.action_space.low.item()
    upper_bound = env.action_space.high.item()
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    tf.random.set_seed(seed)

    actor_model = Actor(action_shape[0], lower_bound, upper_bound)
    actor_model.build((1,) + observation_shape)

    critic_model = get_critic_model(observation_shape[0])
    buffer = EpisodicBuffer()

    episodes_reward = deque(maxlen=min_episodes_criterion)
    writer = tf.summary.create_file_writer(f"runs/{algorithm}-{today}")

    with tqdm.trange(max_episodes) as t:
        for i in t:
            buffer.reset()
            if i > 0 and i % 5 == 0:
                buffer.learn(actor_model, critic_model, actor_optimizer, critic_optimizer)
                buffer.reset_tf()
            initial_state = env.reset()
            reward = run_episode(initial_state, env, actor_model, buffer, max_step)
            episodes_reward.append(reward)
            running_reward = np.mean(episodes_reward)

            observations0, observations1, actions, returns, advantages = buffer.get_tensor(critic_model)
            buffer.store_tensor(observations0, observations1, actions, returns, advantages)

            t.set_description(f"Episode {i}")
            t.set_postfix_str(f" | episode_reward: {reward:.2f}, running_reward: {running_reward:.2f}")
        
            with writer.as_default(step=i):
                tf.summary.scalar(f"Avg. last {min_episodes_criterion} episode reward", running_reward)

            if best_reward < int(running_reward):
                best_reward = int(running_reward)
                sign = "m" if np.sign(best_reward) < 0 else ""
                actor_model.save_weights(f"checkpoints/{algorithm}-{env_id}/reward-{sign}{abs(best_reward)}")

    actor_model.save_weights(f"checkpoints/{algorithm}-{env_id}/latest")
    env.close()

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


def save_video():
    from gym.wrappers import Monitor

    algorithm = "PPO"
    env_id = "Pendulum-v0"

    env = gym.make(env_id)
    env = Monitor(env, f"videos/{algorithm}-{env_id}", force=True)

    lower_bound = env.action_space.low.item()
    upper_bound = env.action_space.high.item()
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    model = Actor(action_shape[0], lower_bound, upper_bound)
    model.build((1,) + observation_shape)

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