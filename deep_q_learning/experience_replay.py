import numpy as np
from collections import deque


class ExperienceReplay(object):
    def __init__(self, memory_size=int(1e+6), batch_size=32):
        self._state = deque([], maxlen=memory_size)
        self._action = deque([], maxlen=memory_size)
        self._state_next = deque([], maxlen=memory_size)
        self._reward = deque([], maxlen=memory_size)
        self._done = deque([], maxlen=memory_size)
        self._batch_size = batch_size
    
    def put(self, state, action, state_next, reward, done):
        self._state.append(state)
        self._action.append(action)
        self._state_next.append(state_next)
        self._reward.append(reward)
        self._done.append(done)

    def get(self):
        indices = np.random.choice(len(self._done), size=self._batch_size)
        state_sample = np.stack([self._state[i] for i in indices])
        action_sample = np.stack([self._action[i] for i in indices])
        state_next_sample = np.stack([self._state_next[i] for i in indices])
        rewards_sample = np.stack([self._reward[i] for i in indices])
        done_sample = np.stack([self._done[i] for i in indices])
        return state_sample, action_sample, state_next_sample, rewards_sample, done_sample

    def get_history_length(self):
        return len(self._done)
