import numpy as np
from collections import deque


class ActorReplayBuffer:
    def __init__(self, trace_length=80, replay_period=40):
        self.trace = deque([], maxlen=trace_length)
        self.period = deque([], maxlen=replay_period)
    
    def __call__(self, *args):
        self.period.append(args)
        if len(self.period) == self.period.maxlen:
            self.trace.append(tuple(self.period))
        if len(self.trace) == self.trace.maxlen:
            return tuple(self.trace)
        else:
            return None


class ReplayBuffer:
    def __init__(self, priority_exponent=0.9, capacity=5000000, minimum_sequences=6250) -> None:
        self.priority_exponent = priority_exponent
        self.minimum_sequences = minimum_sequences
        self.buffer = deque([], maxlen=capacity)

    def push(self, sequences):
        self.buffer.append(sequences)
    
    def sample(self, batch_size=64):
        if len(self.buffer) < self.minimum_sequences:
            return None

        batch_indices = np.random.choice(len(self.buffer), size=batch_size)
        batch = [ self.buffer[index] for index in batch_indices ]

        batch_observations = []
        batch_actions = []
        batch_rewards = []
        batch_discount_factors = []
        batch_intrinsic_rewards = []

        for sequence in batch:
            seq_observation, seq_action, seq_reward, seq_discount_factor, seq_intrinsic_reward = self.rollout(sequence)
            batch_observations.append(seq_observation)
            batch_actions.append(seq_action)
            batch_rewards.append(seq_reward)
            batch_discount_factors.append(seq_discount_factor)
            batch_intrinsic_rewards.append(seq_intrinsic_reward)
            
        return batch_observations, batch_actions, batch_rewards, batch_discount_factors, batch_intrinsic_rewards

    def rollout(self, sequence):
        seq_observation = []
        seq_action = []
        seq_reward = []
        seq_discount_factor = []
        seq_intrinsic_reward = []

        for observation, action, reward, discount_factor, intrinsic_reward in sequence:
            seq_observation.append(observation)
            seq_action.append(action)
            seq_reward.append(reward)
            seq_discount_factor.append(discount_factor)
            seq_intrinsic_reward.append(intrinsic_reward)
        
        return seq_observation, seq_action, seq_reward, seq_discount_factor, seq_intrinsic_reward