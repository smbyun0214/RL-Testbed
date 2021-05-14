import gym
import numpy as np
from collections import deque


class NoopResetEnv(gym.Wrapper):
    """No-op 행동으로 무작위 초기 상태를 샘플링한다.
    No-op는 action 0이라고 가정한다.
    """
    
    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """[1, noop_max] 범위에서 샘플링하여, no-op action을 반복한다."""
        self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    """Life가 줄어들 때마다 episode가 종료되고, 완전히 게임이 끝날 경우에만 reset을 한다."""

    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True

    def reset(self, **kwargs):
        """Life가 없을 경우만 reset을 한다.
        이렇게 하면, life가 있을 때의 모든 state를 만날 수 있게 된다.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # action 0: 어떤 행동도 하지 않는다.
            obs, _, _, _ = self.env.step(0)
        # 현재 life를 저장하고.
        # 만약 reset이 된다면 life가 초기화 된다.
        self.lives = self.env.unwrapped.ale.lives()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # 현재 life가 줄어들었는지 확인한다.
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            done = True
        self.lives = lives
        return obs, reward, done, info


class FireResetEnv(gym.Wrapper):
    """Reset이 될 경우, fire action을 수행해서 게임이 시작되도록 한다."""

    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        return obs


class SkipEnv(gym.Wrapper):
    """`skip`번째 frame을 반환한다."""
    
    def __init__(self, env, skipframe=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skipframe

    def step(self, action):
        """action을 `skip`번 반복하면서 reward를 더한다.
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class FrameStack(gym.Wrapper):
    """가장 최근 k개의 frame을 쌓는다."""

    def __init__(self, env, stackframe):
        super(FrameStack, self).__init__(env)
        self.k = stackframe
        self.frames = deque([], maxlen=stackframe)

        shp = env.observation_space.shape
        shape = (shp[:-1] + (shp[-1]*stackframe,))  # (84, 84, k) = (84, 84) + (1*k, )
        self._shape = shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        self.frames.append(ob)
        for _ in range(self.k - 1):
            ob, _, _, _ = self.env.step(0)
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        frames = np.stack(self.frames, axis=-1)
        return frames.reshape(self._shape)
