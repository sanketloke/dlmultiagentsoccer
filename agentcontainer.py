import random, itertools
from hfo import *
from scripts.environment import SoccerEnvironment
import time
from utils import *
from agents.globals import MyGlobals
import Queue as Q
import copy
from pdb import set_trace as bp
def printQueue(q):
    while not q.empty():
        print (q.get()),
    print ''


def debug():
    print 'Entering debug'




class History(object):

    """Container for storing collection of states."""
    def __init__(self,size):
        super(History, self).__init__()
        self.store=[]
        self.size=size

    def addState(state):
        self.store.append()
        self.checkConditionAndModify()

    def checkConditionAndModify():
        if len(self.store)>self.size:
            self.store= self.store[len(self.store)-self.size:len(self.store)]

    def getStates(k):
        return self.store[len(self.store)-k:len(store)]


class AgentContainer(object):

    """docstring for AgentContainer."""
    def __init__(self, agent,id,teamSize,opponentTeam,queue,rewardargs, featuretransformargs):
        super(AgentContainer, self).__init__()
        self.agent = agent
        self.id =id
        self.teamSize=teamSize
        self.opponentTeam=opponentTeam
        self.q=queue
        self.rewardargs =rewardargs
        self.featuretransformargs = featuretransformargs #May include flush rates

    def run(self):
        print '-----------------------Starting Agent with ID:'+str(self.id)
        hfo = HFOEnvironment()
        print MOVE
        print SHOOT
        print PASS
        print DRIBBLE
        print CATCH
        # Connect to the server with the specified
        # feature set. See feature sets in hfo.py/hfo.hpp.
        time.sleep(1)
        hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
        'bin/teams/base/config/formations-dt', 6000,
        'localhost', 'base_left', False)

        print '-----------------------Connection Successful Agent with ID:'+str(self.id)
        for episode in xrange(100):
            print '-----------------------INSIDE EPISODE Agent with ID:'+str(self.id)
            status = IN_GAME
            # If status is IN_GAME continue
            while status == IN_GAME:


                # Get the vector of state features for the current state
                state = hfo.getState()

                #Adding into the queue for synchronization
                self.q.put(state)
                while (self.q.qsize())<self.teamSize:
                    time.sleep(0)

                history = History()
                #From Queue , get states from other agents
                stateCollection=list(self.q.queue)

                #Preprocess State
                agentState,teamState,opponentState= self.transformState(state,stateCollection,[])

                #Add the current state to history
                self.history.addState([agentState,teamState,opponentState])

                # Update Parameters of the agent
                self.agent.perceive(agentState,teamState,opponentState,reward)


                # Calculate reward for the last action
                reward = calculateReward(history.getStates(3),agentState,teamState,opponentState,[])


                # # Predict Action to be performed
                action,t =self.agent.getAction(state)

                # Perform the Action
                if action==MOVE:
                    hfo.act(MOVE)
                elif action==SHOOT:
                    hfo.act(SHOOT)
                elif action==PASS:
                    hfo.act(PASS,t)
                elif action==DRIBBLE:
                    hfo.act(DRIBBLE)
                else:
                    action=hfo.act(CATCH)

                # Advance the environment and get the game status
                self.q.put(1)
                if self.q.qsize()==2*self.teamSize:
                    with self.q.mutex:
                        self.q.queue.clear()
                status = hfo.step()
                # Check the outcome of the episode

            # Check how the episode ended: if with a goal add to the reward
            # if outside ball, or otherwise add huge penalty.

            print(' ID : '+ str(self.id) +' Episode %d ended with %s'%(episode, statusToString(status)))
            # Quit if the server goes down
            if status == SERVER_DOWN:
                hfo.act(QUIT)
                break
        self.end()


    def end(self):
        print 'Ending Agent '+str(self.id)



    def calculateReward(historyTuple,agentState,teamState,opponentState,rewardargs):
        print 'Custom reward Function'
        reward =0
        reward += -1
        posessionreward= -3 * if agentState[5] == 0 else 0 # Take into account goal proximity
        #Check if pass successful using historyTuple or may be find an alternative way of keeping track of successful passes
        # passingreward=




    """Takes as input: last k states, currentState, rewardargs
      Transform State Function returns 3 state vectors
      Agent State (Size of 9 + T) -> [Features 1-9 described in the manual and passing angles to each of the opponents]
      Team State (Size of (9 + T ) *T-1 )-> Simply concatenating agent state vectors of the teammates
      Opponent State ( 3*O ) -> Vector contains history information of each of the opponents information """
    def transformState(self,rawAgentState,stateCollection,featuretransformargs):
        print 'Transform state'
        agentState= np.append(rawAgentState[0:10] ,rawAgentState [10+2*(self.teamSize-1):10+3*(self.teamSize-1)])
        print "Agent State: "+str(agentState)
        teamState=[]
        #Weird Synchronization fix
        stateCollection = [x for x in stateCollection if type(x)!=type(1)]
        for u in range(len(stateCollection)):
            try:
                if stateCollection[u][0]!=agentState[0] and stateCollection[u][1]!=agentState[1]:
                    teamState +=  stateCollection[u][0:10].tolist()
            except:
                print stateCollection
                print u
                print stateCollection[u]
                bp()
        opponentState= rawAgentState [10+6*(self.teamSize-1)-1:-1]
        return agentState,teamState,opponentState
