

class State:
    def __init__(self, state_coordinate, reward, terminal=False):
        self.state_coordinate = state_coordinate
        self.value = 0.
        self.q_table = {"L": 0, "R": 0, "U": 0, "D": 0}
        self.reward = reward
        self.terminal = terminal
