import argparse
import numpy as np
import tensorflow as tf

from datetime import datetime
from collections import deque

from wrappers.breakout import wrapper
from policies import epsilon_greedy_policy
from logger import Logger


parser = argparse.ArgumentParser(prog="Training")

# name
parser.add_argument('name', type=str, choices=['dqn', 'double_dqn'])

# learning
parser.add_argument('--minibatch_size', type=int, default=32)
parser.add_argument('--update_frequency', type=int, default=4)
parser.add_argument('--discount_factor', type=float, default=0.99)
parser.add_argument('--optimizer', type=str, default="adam")
parser.add_argument('--learning_rate', type=float, default=0.0001)

# epsilon
parser.add_argument('--epsilon', type=float, default=1.0)
parser.add_argument('--epsilon_min', type=float, default=0.1)
parser.add_argument('--epsilon_max', type=float, default=1.0)
parser.add_argument('--epsilon_greedy_frames', type=float, default=500000.0)

# agent
parser.add_argument('--agent_history_length', type=int, default=4)
parser.add_argument('--action_repeat', type=int, default=4)

# experience replay
parser.add_argument('--replay_memory_size', type=int, default=500000)
parser.add_argument('--replay_start_size', type=int, default=50000)

# update target
parser.add_argument('--target_network_update_frequency', type=int, default=50000)

# episode
parser.add_argument('--max_frame_count', type=int, default=10000000)
parser.add_argument('--now', type=str, default=datetime.now().strftime("%Y%m%d-%H%M%S"))
parser.add_argument('--save_log_dir', dest='log_dir', type=str, default="runs")
parser.add_argument('--save_weights_dir', dest='weights_dir', type=str, default="checkpoints")

config = parser.parse_args()

frame_count = 0
episode_count = 0
config.epsilon_interval = config.epsilon_max - config.epsilon_min
config.log_dir += "/{}-{}".format(config.name, config.now)
config.weights_dir += "/{}-{}".format(config.name, config.now)


# Environment
env = wrapper(
    skip=config.action_repeat,
    stack=config.agent_history_length)
num_actions = env.action_space.n

# Agent history
state_history       = deque(maxlen=config.replay_memory_size)
action_history      = deque(maxlen=config.replay_memory_size)
rewards_history     = deque(maxlen=config.replay_memory_size)
state_next_history  = deque(maxlen=config.replay_memory_size)
done_history        = deque(maxlen=config.replay_memory_size)

# Model
assert config.name in ("dqn", "double_dqn")
if config.name == "dqn":
    from algorithms import dqn as train
    from models import create_dqn_model as create_model
elif config.name == "double_dqn":
    from algorithms import double_dqn as train
    from models import create_dqn_model as create_model
    
model               = create_model(num_actions)
model_target        = create_model(num_actions)
model_target.set_weights(model.get_weights())

# optimizer
assert config.optimizer == "adam"
if config.optimizer == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)


# Loger
logger = Logger(config.log_dir)
is_gpu = tf.config.list_physical_devices('GPU')

metric_rewards  = logger.add_metric("Sum. Rewards", "sum", "Sum/Rewards")
metric_loss     = logger.add_metric("Avg. Loss", "mean", "Average/Loss")
metric_q_values = logger.add_metric("Avg. Q-value", "mean", "Average/Q-Values")


# 에피소드 메인 loop
frame_count = 0
episode_count = 0
update_frequency_count = 0
epsilon = config.epsilon_max

while frame_count < config.max_frame_count:    
    # 환경 초기화
    done = False
    state = env.reset()

    # 게임 시작 loop
    while done is False:
        # experience memory에 충분한 경험이 쌓일 때까지 무작위 행동 선택
        if frame_count < config.replay_start_size:
            action = epsilon_greedy_policy(model, state, 1)
        # 충분한 경험이 쌓였을 경우, epsilon greedy로 행동 선택
        else:
            action = epsilon_greedy_policy(model, state, epsilon)
            epsilon -= config.epsilon_interval / config.epsilon_greedy_frames
            epsilon = max(epsilon, config.epsilon_min)

        # 선택한 행동에 대한 environment의 응답
        state_next, reward, done, info = env.step(action)

        # observable 저장
        state_history.append(state)
        action_history.append(action)
        rewards_history.append(reward)
        state_next_history.append(state_next)
        done_history.append([ 1 if done else 0])

        # experience memory에 충분한 경험이 쌓일 때 학습 진행
        if len(done_history) > config.replay_start_size:
            # minibatch_size 만큼 sample을 생성하여 학습을 진행한다.
            # update_frequency 반복한다.
            for _ in range(config.update_frequency):
                indices = np.random.choice(range(len(done_history)), size=config.minibatch_size)

                state_sample        = np.stack([state_history[i]        for i in indices])
                action_sample       = np.stack([action_history[i]       for i in indices])
                rewards_sample      = np.stack([rewards_history[i]      for i in indices])
                state_next_sample   = np.stack([state_next_history[i]   for i in indices])
                done_sample         = np.stack([done_history[i]         for i in indices])

                loss, q_max = train(
                    model,
                    model_target,
                    optimizer,
                    config.discount_factor,
                    num_actions,
                    state_sample, action_sample, rewards_sample, state_next_sample, done_sample)

                update_frequency_count += 1
                metric_loss(loss)
                metric_q_values(q_max)
                
            # target model을 업데이트하면서, 저장한다.
            if update_frequency_count % config.target_network_update_frequency == 0:
                model_target.set_weights(model.get_weights())
                model.save_weights(config.weights_dir)

        # 게임 점수를 더한다.
        metric_rewards(reward)

        # 에피소드가 끝나지 않을 경우, 다음 진행
        state = state_next
        frame_count += 1

    # 로그를 기록하고 출력한다.
    episode_count += 1
    print("Episode: {} | Loss: {:.4f} | Q-value: {:.4f} | Rewards: {} | epsilon: {:.4f} | memory size: {:}".format(
        episode_count, metric_loss.result(), metric_q_values.result(), metric_rewards.result(), epsilon, len(done_history))
    )
    logger.write_metrics(episode_count)
    logger.write_scalar(epsilon, "Etc./Epsilon", episode_count)
    if is_gpu:
        logger.write_scalar(
            tf.config.experimental.get_memory_usage("GPU:0"),
            "Etc./GPU usages",
            episode_count
        )
    logger.reset_metrics()
