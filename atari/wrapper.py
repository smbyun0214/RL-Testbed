import gym
from atari.environment import NoopResetEnv
from atari.environment import EpisodicLifeEnv
from atari.environment import FireResetEnv
from atari.environment import SkipEnv
from atari.environment import FrameStack
from atari.observation import FlickerFrame 
from atari.observation import WarpFrame

def wrapper(env_id, skip=4, stack=4, filcker_probability=None, seed=None):
    env = gym.make(env_id)
    env = NoopResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    if skip > 1:
        env = SkipEnv(env, skip)
    if filcker_probability:
        env = FlickerFrame(env, filcker_probability)
    if stack > 1:
        env = FrameStack(env, stack)
    if seed:
        env.seed(seed)
    return env
