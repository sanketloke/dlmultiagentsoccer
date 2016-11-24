import random, itertools
from hfo import *
from scripts.environment import SoccerEnvironment
import time
from utils import *
from agents.randomagent import RandomAgent
from agents.globals import MyGlobals


teamSize =3
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


#Input Vector: 11 + (teamSize -1)+(teamSize -1)*11 +3 * opponentsSize



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

#startPool(agentContainers)


# Iterate till total no of episodes

print 'Hello'
# for i in range(total_episodes):
# 	Observe inputs from all agents
#	Wait till we receive from all of them
#	Aggregate them
#	Pass the aggregated inputs to the agents, compute the actions
#

# Iterate ends








import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

#from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from agents.dqn import DQNAgent


#from pudb import set_trace; set_trace()

# Get the environment and extract the number of actions.
nb_actions = 3+(teamSize-1)
environmentVectorSize=10 + (teamSize -1)*11 +3 * opponentsSize
print environmentVectorSize
# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + (environmentVectorSize,)))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
# dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
#                target_model_update=1e-2, policy=policy)
#dqn.compile(Adam(lr=1e-3), metrics=['mae'])

import Queue as Q
agents=[]
q=Q.Queue()
u2=team[:]
for i in team:
    d= DQNAgent( i,u2,opponents,actions,actionsEnum,model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,target_model_update=1e-2, policy=policy, )
    d.compile(Adam(lr=1e-3), metrics=['mae'])
    agents.append(d)

agentContainers=[]
for i in agents:
    agentContainers.append(AgentContainer(i,i.id,teamSize,opponentsSize,q,[],[]))



startPool(agentContainers)






























# Join Environment thread
env.end()


# Display Statistics
