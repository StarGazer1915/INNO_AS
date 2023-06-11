from random import sample
from collections import deque


class Memory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.deque = deque()

    def store(self, transition):
        """
        Stores new transitions in the memory.
        @return: void
        """
        if len(self.deque) >= self.memory_size:
            remove_first = self.deque.popleft()
            self.deque.append(transition)
        else:
            self.deque.append(transition)

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
