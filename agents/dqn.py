from baseagent import Agent

# Should have a network attribute
class DQNAgent(Agent):

    """docstring for DQNAgent."""
    def __init__(self, arg):
        super(DQNAgent, self).__init__()
        self.arg = arg

    def getQUpdate(self,args):
        print "Calculate Q values"

    def nql:qLearnMinibatch(self,args):
        print "Perform a minibatch Q-learning update"

    def perceive(agentState,teamState,opponentState,reward):
        print "perceive function"

    def eGreedy(self):
        print "Epsilon Greedy for exploration"

    def greedy(self):
        print "Exploration"

    def getAction():
        print "Returns Predicted Action"
