

class State:
    def __init__(self, state_coordinate, reward, terminal=False):
        self.state_coordinate = state_coordinate
        self.value = 0.
        self.q_table = {}
        self.reward = reward
        self.terminal = terminal
