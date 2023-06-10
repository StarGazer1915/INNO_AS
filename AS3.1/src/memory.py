from random import sample


class Memory:
    def __init__(self):
        self.deque = []

    def store(self):
        """
        Stores new transitions in the memory.
        @return: void
        """
        return

    def sample(self, sample_size):
        """
        Takes a batch of transitions out of the memory.
        @return: void
        """
        if len(self.deque) >= sample_size:
            choices = sample(range(0, len(self.deque)), sample_size)
        elif sample_size < 0:
            return []
        else:
            choices = sample(range(0, len(self.deque)), len(self.deque))
        return [self.deque[:][i] for i in choices]
