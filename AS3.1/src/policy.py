

class Policy:
    def __init__(self, func_approx, epsilon):
        self.neural_net = func_approx
        self.epsilon = epsilon

    def select_action(self):
        """
        Chooses an action based on a given state.
        @return: void
        """
        return

    def decay(self):
        """
        Degrades epsilon over time.
        @return: void
        """
        return