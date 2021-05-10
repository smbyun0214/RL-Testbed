import argparse
from datetime import datetime
from common import load_pickle
from common import save_pickle


parser = argparse.ArgumentParser(prog="Reinforcement Learning Testbed")

# setup parameters
parser.add_argument('name', type=str, choices=['dqn', 'double_dqn'])
parser.add_argument('solved_reward', type=int)

# hyperparameters
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--minibatch_size', dest="batch_size", default=32, type=int)
parser.add_argument('--replay_memory_size',dest="replay_size", default=1000000, type=int)
parser.add_argument('--replay_start_size', dest="replay_start_size", default=50000, type=int)
parser.add_argument('--agent_history_length', dest="stackframe", default=4, type=int)
parser.add_argument('--action_repeat', dest="skipframe", default=4, type=int)
parser.add_argument('--discount_factor', default=0.99, type=float)
parser.add_argument('--update_frequency', dest="parameter_update_frequency", default=4, type=int)
parser.add_argument('--target_network_update_frequency', dest="model_target_update_frequency", default=10000, type=int)
parser.add_argument('--optimizer', default="adam", choices=["adam"], type=str)
parser.add_argument('--learning_rate', default=0.00001, type=float)
parser.add_argument('--initial_exploration', dest="epsilon", default=1.0, type=float)
parser.add_argument('--final_exploration', dest="epsilon_min", default=0.1, type=float)
parser.add_argument('--final_exploration_frame', dest="epsilon_frame", default=1000000.0, type=float)

# save & load
parser.add_argument('--now', default=datetime.now().strftime("%Y%m%d-%H%M%S"), type=str)
parser.add_argument('--log_dir', default="runs", type=str)
parser.add_argument('--weights_dir', default="checkpoints", type=str)
parser.add_argument('--load_config_file', default="", type=str)
parser.add_argument('--save_config_file', default="", type=str)

# setup config
config = parser.parse_args()

time_tmpl ="%Y%m%d-%H%M%S" 
log_dir_tmpl = "{}/{}-{}".format(config.log_dir, config.name, {})
weights_dir_tmpl = "{}/{}-{}".format(config.weights_dir, config.name, {})
now = datetime.now().strftime(time_tmpl)

# load config
if config.load_config_file:
    config = load_pickle(config.load_config_file)
    assert config.lastest_weights_file
    assert config.update_count

# importer
import numpy as np
import tensorflow as tf
from collections import deque
from wrappers.pong import wrapper
from policies import epsilon_greedy_policy
from agent import Agent
from experience_replay import ExperienceReplay
from logger import Logger

if config.name in ("dqn", "double_dqn"):
    from models import create_dqn_model as create_model
if config.name == "dqn":
    from algorithms import dqn as train
elif config.name == "double_dqn":
    from algorithms import double_dqn as train

# environment
env = wrapper(skip=config.skipframe, stack=config.stackframe)
num_actions = env.action_space.n

# set random seed
np.random.seed(42)
tf.random.set_seed(42)
env.seed(config.seed)

# model
model = create_model(num_actions)
if config.load_config_file:
    model = model.load_weights(config.lastest_weights_file)

# agent
agent = Agent(
    model,
    num_actions,
    epsilon_greedy_policy,
    train,
    epsilon_init=config.epsilon,
    epsilon_fin=config.epsilon_min,
    epsilon_fin_frame=config.epsilon_frame,
    learning_rate=config.learning_rate
)

# experience replay
buffer = ExperienceReplay(memory_size=config.replay_size, batch_size=config.batch_size)

# logger
logger = tf.summary.create_file_writer(log_dir_tmpl.format(now))
is_gpu = tf.config.list_physical_devices('GPU')

# metrics
metric_q = tf.keras.metrics.Mean("Avg. Q-value", dtype=tf.float32)
metric_loss = tf.keras.metrics.Mean("Avg. Loss", dtype=tf.float32)
metric_rewards = tf.keras.metrics.Sum("Sum. Rewards", dtype=tf.float32)


# main
all_rewards = deque([], maxlen=100)

if config.load_config_file:
    update_count = config.update_count
else:
    update_count = 0

frame_count = 0

while True:    
    done = False
    state = env.reset()

    while done is False:
        if frame_count < config.replay_start_size:
            action = agent.get_action(state, 1.0)
        else:
            action = agent.get_action(state)

        state_next, reward, done, info = env.step(action)
        buffer.put(state, [action], state_next, [reward], [done])

        if buffer.get_history_length() >= config.replay_start_size:
            for _ in range(config.parameter_update_frequency):
                state_sample, action_sample, state_next_sample, rewards_sample, done_sample = buffer.get()
                loss, q_max = agent.train(state_sample, action_sample, state_next_sample, rewards_sample, done_sample)
                metric_loss(loss)
                metric_q(q_max)
                update_count += 1

                with logger.as_default():
                    tf.summary.scalar("Performance/Q-value", metric_q.result(), step=update_count)
                    tf.summary.scalar("Performance/Loss", metric_loss.result(), step=update_count)
                    if is_gpu:
                        tf.summary.scalar(
                            "Etc./GPU usages",
                            tf.config.experimental.get_memory_usage("GPU:0"),
                            step=update_count
                        )
            
                if update_count % config.model_target_update_frequency == 0:
                    agent.update_model_target()
                    agent.save_model(weights_dir_tmpl.format(now))

                    if config.save_config_file:
                        config.update_count = update_count
                        config.epsilon = agent.epsilon
                        config.replay_start_size = buffer.get_history_length()
                        save_pickle(config.save_config_file, config)

        metric_rewards(reward)
        state = state_next
        frame_count += 1

    with logger.as_default():
        tf.summary.scalar("Performance/Rewards", metric_rewards.result(), step=update_count)
        tf.summary.scalar("Etc./Epsilon", agent.epsilon, step=update_count)

    print("{} | Loss: {:.4f} | Q-value: {:.4f} | Rewards: {:4f} | epsilon: {:.4f} | memory size: {:}".format(
        update_count, metric_loss.result(), metric_q.result(), metric_rewards.result(), agent.epsilon, buffer.get_history_length())
    )

    all_rewards.append(metric_rewards.result())
    if np.mean(all_rewards) >= config.solved_reward:
        agent.save_model(weights_dir_tmpl.format("done"))
        break

    metric_q.reset_states()
    metric_loss.reset_states()
    metric_rewards.reset_states()
