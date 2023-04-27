from random import choice


class Policy:
    def __init__(self, policy_type="", gamma=1.):
        self.type = policy_type
        self.gamma = gamma

    def select_action(self, actions):
        if self.type.lower() == "random":
            return choice(actions)
        else:
            return choice(actions)
