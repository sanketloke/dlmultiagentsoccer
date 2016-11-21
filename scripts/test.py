from environment import SoccerEnvironment

args={}
args['offense-agents']=1
args['defense-agents']=0
args['defense-npcs']=2
args['offense-npcs']=2
k=SoccerEnvironment(args)
k.start()


import time
time.sleep(100) # delays for 5 seconds


k.end()
