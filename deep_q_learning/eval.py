import time
from wrappers.breakout import wrapper
from models import create_dqn_model as create_model
from policies import epsilon_greedy_policy

# name
name = "dqn-model"
frame_count = "20210504-071205_2130000"

# agent
action_repeat = 4
agent_history_length = 4

# Environment
env = wrapper(skip=action_repeat, stack=agent_history_length)
num_actions = env.action_space.n

# model save
model_name_temp = "checkpoints/{}-{}"

# Model
model = create_model(num_actions)
model.load_weights(model_name_temp.format(name, frame_count))

try:
    while True:
        state = env.reset()
        done = False
        i = 0
        reward = 0
        while done is False:
            i += 1
            env.render()
            action = epsilon_greedy_policy(model, state, 0)
            state, _reward, done, _ = env.step(action)
            # import matplotlib.pyplot as plt
            # _, axes = plt.subplots(nrows=1, ncols=4)
            # axes[0].matshow(state[:, :, 0])
            # axes[1].matshow(state[:, :, 1])
            # axes[2].matshow(state[:, :, 2])
            # axes[3].matshow(state[:, :, 3])
            # plt.show()
            if done:
                _reward = -1
            time.sleep(.05)
            reward += _reward
        print(i, reward, reward / i)
except:
    pass
finally:
    env.close()

