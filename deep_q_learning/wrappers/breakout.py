import gym
from wrappers.environment import NoopResetEnv
from wrappers.environment import EpisodicLifeEnv
from wrappers.environment import FireResetEnv
from wrappers.environment import SkipEnv
from wrappers.environment import FrameStack
from wrappers.observation import WarpFrame

def wrapper(env_id="BreakoutNoFrameskip-v4", skip=4, stack=4):
    env = gym.make(env_id)
    env = NoopResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = SkipEnv(env, skip=skip)
    env = FrameStack(env, k=stack)
    return env
