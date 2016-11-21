#multitask: 1. get mixture weights from input; 2. opponent prediction
class DQNOpponentMultiTaskGroupAgent(DQNOpponentOpponentAgent):

    """docstring for DQNOpponentMultiTaskGroupAgent."""
    def __init__(self, arg):
        super(DQNOpponentMultiTaskGroupAgent, self).__init__()
        self.arg = arg
