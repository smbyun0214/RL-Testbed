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


class DRQNExperienceReplay(object):
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

    def get(self, num_timesteps=10):
        state_sample = []
        action_sample = []
        state_next_sample = []
        rewards_sample = []
        done_sample = []
        while len(done_sample) < self._batch_size:
            index = np.random.randint(len(self._done)-num_timesteps)
            _state_sample = []
            _action_sample = [] 
            _state_next_sample = []
            _rewards_sample = []
            _done_sample = []
            for step in range(num_timesteps):
                next_step = index + step
                _state_sample.append(self._state[next_step])
                _action_sample.append(self._action[next_step])
                _state_next_sample.append(self._state_next[next_step])
                _rewards_sample.append(self._reward[next_step])
                _done_sample.append(self._done[next_step])
                if step+1 == num_timesteps:
                    _action_sample = _action_sample[-1]
                    _rewards_sample = np.sum(_rewards_sample, axis=0)
                    _done_sample = _done_sample[-1]
                elif self._done[next_step] == [True]:
                    break
            else:
                state_sample.append(_state_sample)
                action_sample.append(_action_sample)
                state_next_sample.append(_state_next_sample)
                rewards_sample.append(_rewards_sample)
                done_sample.append(_done_sample)
        state_sample = np.stack(state_sample)
        action_sample = np.stack(action_sample)
        state_next_sample = np.stack(state_next_sample)
        rewards_sample = np.stack(rewards_sample)
        done_sample = np.stack(done_sample)
        return state_sample, action_sample, state_next_sample, rewards_sample, done_sample

    def get_history_length(self):
        return len(self._done)
