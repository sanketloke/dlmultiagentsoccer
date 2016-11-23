import random, itertools
from hfo import *
from scripts.environment import SoccerEnvironment
import time
from utils import *
from agents.randomagent import RandomAgent
from agents.globals import MyGlobals


teamSize =6
opponentsSize=2
# Define  Environment Configuration
envargs={}
envargs['offense-agents']=teamSize
envargs['defense-agents']=0
envargs['defense-npcs']=opponentsSize
envargs['offense-npcs']=0
# TODO support gui-less environment
env = SoccerEnvironment(envargs)


#Initialize Individual Agents




MyGlobals.stateVec=[]

#Initialize Environment and launch a thread
env.start()
time.sleep(1)
print 'Connecting to Environment'


#Defining actions
actions = [int(MOVE) , int(SHOOT), int(PASS), int(DRIBBLE),int(CATCH)]
actionsEnum = {8:MOVE, 9:SHOOT, 10:PASS,11:DRIBBLE,12:CATCH}




from agentcontainer import AgentContainer
from multiprocessing.pool import ThreadPool
import threading
from threading import Thread

team=[ i+1 for i in range(teamSize)]
opponents = [ i+1 for i in range(opponentsSize)]


import Queue as Q
agents=[]
q=Q.Queue()
u2=team[:]
for i in team:
    agents.append(RandomAgent( i,u2,opponents,actions,actionsEnum))

agentContainers=[]
for i in agents:
    agentContainers.append(AgentContainer(i,i.id,teamSize,opponentsSize,q,[],[]))
print len(agentContainers)
# a1= AgentContainer(RandomAgent(),1)
# a2 = AgentContainer([],2)
# a3 = AgentContainer([],3)
# a4 = AgentContainer([],4)
# a5 = AgentContainer([],5)
#

def startThread(a):
    a.run()


def startPool(agents):
    t=[]
    for i in agents:
        tmp= Thread(target = startThread, args =(i,))
        t.append(tmp)

    for i in t:
        i.start()
        time.sleep(1)
        print "STARTING----------------------------------------------------------------------------------------"

    for i in t:
        i.join()


#a=[a1,a2,a3,a4,a5]

startPool(agentContainers)


# Iterate till total no of episodes

print 'Hello'
# for i in range(total_episodes):
# 	Observe inputs from all agents
#	Wait till we receive from all of them
#	Aggregate them
#	Pass the aggregated inputs to the agents, compute the actions
#

# Iterate ends








# Join Environment thread
env.end()


# Display Statistics
