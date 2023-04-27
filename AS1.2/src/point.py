

class Point:
    def __init__(self, state_coordinate, reward, terminal=False):
        self.state_coordinate = state_coordinate
        self.value = 0.
        self.reward = reward
        self.terminal = terminal
