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

@tf.function
def update_actor_model(
    actor_model: tf.keras.Model,
    flatten: tf.Tensor):
    start = 0
    for var in actor_model.trainable_variables:
        step = tf.size(var)
        new_var = tf.reshape(flatten[start:start+step], var.shape)
        var.assign(new_var)
        start += step
        

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
    
    def remember(self, observation0: np.ndarray, action: np.ndarray, reward: np.ndarray, observation1: np.ndarray):
        self.observations0.append(observation0)
        self.observations1.append(observation1)
        self.actions.append(action)
        self.rewards.append(reward)

    # @tf.function
    def store_tensor(self, observations0, observations1, actions, returns):
        self.tf_observations0 = observations0 if self.tf_observations0 is None else tf.concat([self.tf_observations0, observations0], axis=0)
        self.tf_observations1 = observations1 if self.tf_observations1 is None else tf.concat([self.tf_observations1, observations1], axis=0)
        self.tf_actions = actions if self.tf_actions is None else tf.concat([self.tf_actions, actions], axis=0)
        self.tf_returns = returns if self.tf_returns is None else tf.concat([self.tf_returns, returns], axis=0)
    
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

    def get_tensor(self, discount_factor=0.99) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        observations0 = tf.convert_to_tensor(self.observations0, dtype=tf.float32)
        observations1 = tf.convert_to_tensor(self.observations1, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.actions, dtype=tf.float32)
        returns = self.get_expected_return(discount_factor)
        returns = tf.convert_to_tensor(np.expand_dims(returns, 1), dtype=tf.float32)
        
        return (observations0, observations1, actions, returns)

    def learn(self, actor_model: tf.keras.Model, critic_model: tf.keras.Model, critic_optimizer: tf.keras.optimizers.Optimizer, kl_max=0.01, critic_batch_size = 64, critic_iter=5):
        observations0, actions, returns = self.tf_observations0, self.tf_actions, self.tf_returns
        self.training_actor(actor_model, critic_model, observations0, actions, returns, kl_max)
        self.training_critic(critic_model, critic_optimizer, observations0, returns, critic_batch_size, critic_iter)

    @tf.function
    def training_actor(self, actor_model: tf.keras.Model, critic_model: tf.keras.Model, observations0: tf.Tensor, actions: tf.Tensor, returns: tf.Tensor, kl_max: float):
        old_dists = get_dists(actor_model, observations0)
        old_dists_fvp = get_dists(actor_model, observations0[::5])

        with tf.GradientTape() as tape:
            values = critic_model(observations0)
            advantage = returns - values
            advantage = (advantage - tf.math.reduce_mean(advantage)) / tf.math.reduce_std(advantage)
            surrogate = get_surrogate_function(actor_model, old_dists, observations0, actions, advantage)

        surrogate_grads = tape.gradient(surrogate, actor_model.trainable_variables)
        surrogate_grads_flatten = flatten(surrogate_grads)

        step_dir = conjugate_gradient(surrogate_grads_flatten, actor_model, old_dists, observations0)

        hs = fisher_vector_product(step_dir, actor_model, old_dists_fvp, observations0[::5])

        shs = tf.math.reduce_sum(step_dir * hs)
        beta = tf.math.sqrt(2 * kl_max / shs)
        full_step = beta * step_dir

        expected_improve = tf.math.reduce_sum(surrogate_grads_flatten * full_step)

        is_update = False
        step_size = 1.0
        theta = flatten(actor_model.trainable_variables)

        for _ in tf.range(10):
            new_theta = theta + full_step*step_size

            update_actor_model(actor_model, new_theta)

            new_surrogate = get_surrogate_function(actor_model, old_dists, observations0, actions, advantage)
            improve = new_surrogate - surrogate
            
            new_dists = get_dists(actor_model, observations0)

            kl = old_dists.kl_divergence(new_dists)
            kl_mean = tf.math.reduce_mean(kl)
            
            if not tf.math.reduce_all(tf.math.is_finite([kl_mean, improve])):
                # tf.print("Got non-finite value of losses -- bad!")
                pass
            elif kl_mean > kl_max * 1.5:
                # tf.print(f"violated KL constraint. shrinking step. kl_mean({kl_mean}) > kl_max({kl_max})")
                pass
            elif improve < 0:
                # tf.print(f"surrogate didn't improve. shrinking step. improve({improve}) < 0")
                pass
            else:
                # tf.print("Stepsize OK!")
                is_update = True
                break
            step_size *= 0.5
        
        if not is_update:
            tf.print("couldn't compute a good step")
            update_actor_model(actor_model, theta)

    @tf.function
    def training_critic(self, critic_model: tf.keras.Model, critic_optimizer: tf.keras.optimizers.Optimizer, observations0: tf.Tensor, returns:tf.Tensor, batch_size: int, iter: int):
        length = returns.shape[0]
        
        for _ in tf.range(iter):
            batch_indices = tf.random.categorical(tf.zeros((1, length)), batch_size)[0]

            batch_observation0 = tf.gather(observations0, batch_indices)
            batch_returns = tf.gather(returns, batch_indices)

            with tf.GradientTape() as tape:
                batch_values = critic_model(batch_observation0)
                critic_loss = self.huber_loss(batch_returns, batch_values)
            
            critic_grads = tape.gradient(critic_loss, critic_model.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_grads, critic_model.trainable_variables))




def env_step(env: gym.Env, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, reward, done, _ = env.step(actions)
    return state.astype(np.float32), np.array(reward, dtype=np.float32), np.array(done, dtype=bool)


def run_episode(
    initial_state: np.ndarray,
    env: gym.Env,
    actor_model: tf.keras.Model,
    critic_model: tf.keras.Model,
    critic_optimizer: tf.keras.optimizers.Optimizer,
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


@tf.function
def flatten(var_list: List[tf.Tensor]) -> tf.Tensor:
    return tf.concat([ tf.reshape(var, [tf.size(var)]) for var in var_list ], axis=0)


@tf.function
def get_surrogate_function(
    actor_model: tf.keras.Model,
    old_dists: tfp.distributions.Normal,
    states: tf.Tensor,
    actions: tf.Tensor,
    advantage: tf.Tensor) -> tf.Tensor:
    new_dists = get_dists(actor_model, states)

    old_log_probs = old_dists.log_prob(actions)
    new_log_probs = new_dists.log_prob(actions)

    surrogate = tf.math.exp(new_log_probs - old_log_probs) * advantage
    return tf.math.reduce_mean(surrogate)


@tf.function
def fisher_vector_product(
    y: tf.Tensor,
    actor_model: tf.keras.Model,
    old_dists: tfp.distributions.Normal,
    states: tf.Tensor,
    cg_damping: float = 0.001) -> tf.Tensor:
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            new_dists = get_dists(actor_model, states)
            kl = old_dists.kl_divergence(new_dists)
        # JM
        kl_grads = t1.gradient(kl, actor_model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        kl_grads_flatten = flatten(kl_grads)

        # JMy
        kl_grads_y = tf.math.reduce_sum(kl_grads_flatten * y)
    
    # M'JMy
    fvp = t2.gradient(kl_grads_y, actor_model.trainable_variables)
    fvp_flatten = flatten(fvp)

    return fvp_flatten + cg_damping * y


@tf.function
def conjugate_gradient(
    b: tf.Tensor,
    actor_model: tf.keras.Model,
    old_dists: tfp.distributions.Normal,
    states: tf.Tensor,
    iter: int = 10,
    residual_tol: float = eps):

    x = tf.zeros_like(b)
    d = tf.identity(b) # b - mA @ x0
    r = tf.identity(b) # b - mA @ x0

    rdotr = tf.math.reduce_sum(r * r)

    for i in tf.range(iter):
        fvp = fisher_vector_product(d, actor_model, old_dists, states)
    
        alpha = rdotr / tf.math.reduce_sum(d * fvp)
        x = x + alpha * d
        r = r - alpha * fvp
        
        new_rdotr = tf.math.reduce_sum(r * r)
        beta = new_rdotr / rdotr
        d = r + beta * d

        rdotr = new_rdotr
        if rdotr < residual_tol:
            break

    return x


def main():
    algorithm = "TRPO"
    env_id = "Pendulum-v0"

    max_episodes = 50000
    max_step = 200
    seed = 42
    min_episodes_criterion = 100
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
                buffer.learn(actor_model, critic_model, critic_optimizer)
                buffer.reset_tf()
            initial_state = env.reset()
            reward = run_episode(initial_state, env, actor_model, critic_model, critic_optimizer, buffer, max_step)
            episodes_reward.append(reward)
            running_reward = np.mean(episodes_reward)

            observations0, observations1, actions, returns = buffer.get_tensor()
            buffer.store_tensor(observations0, observations1, actions, returns)

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

    algorithm = "TRPO"
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