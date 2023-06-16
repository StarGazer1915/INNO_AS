from random import sample
from collections import deque, namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Memory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.deque = deque([], maxlen=memory_size)

    def store(self, state, action, reward, next_state):
        """
        Stores new transitions in the memory.
        @return: void
        """
        self.deque.append(Transition(state, action, reward, next_state))

    def sample(self, sample_size):
        """
        Takes a batch of transitions out of the memory.
        @return: list
        """
        if sample_size <= 0:
            return []
        elif len(self.deque) >= sample_size:
            return sample(self.deque, sample_size)
        else:
            return sample(self.deque, len(self.deque))
