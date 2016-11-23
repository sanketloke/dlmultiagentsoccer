from random import randint
import random
from baseagent import Agent
from hfo import *
class RandomAgent(Agent):

    """docstring for RandomAgent."""
    def __init__(self, id,teammates,opponents,actions,actionsEnum):
        super(RandomAgent, self).__init__()
        self.id=id
        teammates.remove(self.id)
        self.teammates= teammates
        self.opponents= opponents
        self.actions ,self.actionsEnum =actions,actionsEnum
        self.minAct,self.maxAct=min(self.actions),max(self.actions)

    def getAction(self,state):
        t=-1
        if state[5]==1:
            a= (random.choice([SHOOT, DRIBBLE]))
        else:
            a= MOVE
        if a==PASS:
            t= random.choice(self.teammates)
        return (a,t)


    def perceive(self,agentState,teamState,opponentState,reward):
        print "Random Action"
    
