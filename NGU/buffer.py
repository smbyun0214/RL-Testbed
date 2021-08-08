from typing import List, Tuple, Iterable, Optional
import numpy as np
from collections import deque
import itertools
from ext_typing import Observation, Action, Reward, Transition, Trace


class EpisodeBuffer:
    def __init__(self) -> None:
        self.observations = deque() # [Observation]
        self.actions = deque()      # [Action]
        self.rewards = deque()      # [Reward]

    @property
    def length(self) -> int:
        """ 현재 에피소드의 버퍼 길이 """
        return len(self.actions)

    @property
    def is_empty(self) -> bool:
        """ 현재 에피소드가 비어있는지 확인 """
        return len(self.actions) == 0
    
    def clear(self) -> None:
        """ 현재 에피소드의 버퍼 비우기 """
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()

    def push_back(self, transition: Transition) -> None:
        """ 현재 Transition 정보 입력"""
        if self.is_empty:
            self.observations.append(transition.observation)
            self.observations.append(transition.next_observation)
        else:
            assert np.array_equal(self.observations[-1], transition.observation)
            self.observations.append(transition.next_observation)
        self.actions.append(transition.action)
        self.rewards.append(transition.reward)
    
    def pop_front(self) -> None:
        """ 현재 에피소드의 버퍼에 있는 가장 오래된 transition을 제거 """
        if not self.is_empty():
            self.observations.popleft()
            self.actions.popleft()
            self.rewards.popleft()

    def get_random_trace(self, timesteps: int) -> Trace:
        """ timesteps 길이만큼 trace를 생성 """
        assert 0 <= self.length - timesteps < self.length
        index = np.random.choice(self.length - timesteps + 1)
        return Trace(
            list(itertools.islice(self.observations, index, None)),
            list(itertools.islice(self.actions, index, None)),
            list(itertools.islice(self.rewards, index, None)))

    def get_last_transition(self, timesteps: Optional[int] = None) -> Trace:
        """ 가장 최근 timesteps 길이만큼 trace를 생성 """
        if timesteps is None:
            index = self.length - 1
        else:
            index = self.length - timesteps

        assert 0 <= index < self.length
        return Trace(
            list(itertools.islice(self.observations, index, None)),
            list(itertools.islice(self.actions, index, None)),
            list(itertools.islice(self.rewards, index, None)))
    

class ActorBuffer:
    def __init__(self, trace_length=80, replay_period=40):
        self.trace_length = trace_length
        self.replay_period = replay_period

        self.history = deque()              # [EpisodeBuffer]
        self.history_trace_count = deque()  # [EpisodeBuffer의 길이]
        
        self.current = EpisodeBuffer()

    """˝
    Property
    """
    @property
    def sequence_count(self):
        count = sum(self.history_trace_count)
        count += self.current.length // self.trace_length
        return count

    """
    Push/Pop
    """
    def push_back(self, observation: Observation, action: Action, reward: Reward, next_observation: Observation) -> None:
        self.current.push_back(observation, action, reward, next_observation)

    def pop_front(self):
        while len(self.history) > 0:
            if self.history[0].is_empty():
                self.history.popleft()
                self.history.popleft()
                continue
            episode = self.history[0]
            episode.pop_front()
            break

    """
    Done Episode
    """
    def done_episode(self):
        if self.current.length >= self.trace_length:
            self.history.append(self.current)
            self.history_trace_count.append(self.current.length // self.trace_length)

    """
    Generate period/sequence
    """
    def get_last_period(self) -> Transition:
        if self.current.length >= self.replay_period:
            return self.current.get_last_transition(self.replay_period)
        else:
            return self.current.get_empty_transition()

    def get_random_sequence(self):
        episode_count = len(self.history) + 1 if self.current.length >= self.trace_length else 0
        episode_index = np.random.choice(episode_count)
        choose_episode = self.history[episode_index] if episode_index < len(self.history) else self.current
        return choose_episode.get_random_transition(self.trace_length)



class LearnerBuffer:
    def __init__(self, priority_exponent=0.9, capacity=5000000, minimum_sequences=6250) -> None:
        self.priority_exponent = priority_exponent
        self.minimum_sequences = minimum_sequences
        self.actor_buffers = deque()    # [ActorBuffer]

    def add_buffer(self, buffer: ActorBuffer):
        self.actor_buffers.append(buffer)
    
    def get_samples(self, batch_size=64) -> List[Transition]:
        actor_sequence_count = [ buffer.sequence_count for buffer in self.actor_buffers ]
        if sum(actor_sequence_count) < self.minimum_sequences:
            return None

        probabilities = np.array(actor_sequence_count) / sum(actor_sequence_count)
        actor_indices = np.random.choice(len(self.actor_buffers), size=batch_size, p=probabilities)

        batch_transitions = [ self.actor_buffers[index].get_random_sequence() for index in actor_indices ]
        return batch_transitions


if __name__ == "__main__":
    import gym
    import gym_maze
    from wrapper import MyWrapper
    from buffer import ActorBuffer, LearnerBuffer

    env = gym.make("maze-sample-10x10-v0")
    env = MyWrapper(env)

    obs = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, _ = env.step(action)

    actor_buffer = ActorBuffer()
    actor_buffer.push_back(obs, action, reward, next_obs)
    
    # actor_buffer.get_random_sequence()
    # actor_buffer.history[0].get_random_transition(10)
    learner_buffer = LearnerBuffer(minimum_sequences=1)
    learner_buffer.add_buffer(actor_buffer)
    learner_buffer.add_buffer(actor_buffer)
    
    learner_buffer.get_samples(1)
    