from random import choice


class Policy:
    def __init__(self, policy_type="", gamma=1.):
        self.type = policy_type
        self.gamma = gamma

    def select_action(self, actions):
        """
        This function decides the action that the agent is going
        to take within the maze environment.
        @param actions: nested list
        @return: list [y, x]
        """
        if self.type.lower() == "random":
            return choice(actions)
        else:
            return choice(actions)
