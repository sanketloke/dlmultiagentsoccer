from baseagent import Agent
#multitask: 1. get mixture weights from input; 2. opponent prediction
class DQNOpponentMultiTaskActionAgent(DQNOpponentOpponentAgent):
    """docstring for DQNOpponentMultiTaskActionAgent."""
    def __init__(self, arg):
        super(DQNOpponentMultiTaskActionAgent, self).__init__()
        self.arg = arg
