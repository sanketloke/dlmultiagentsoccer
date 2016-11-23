import subprocess
import os
import signal
import time

class SoccerEnvironment(object):

    """docstring for Environment."""
    def __init__(self, arg):
        super(SoccerEnvironment, self).__init__()
        self.arg = arg

    def start(self):
        cmd = ["bin/HFO"]
        if (int(self.arg["offense-agents"])>0):
            cmd.append("--offense-agents="+str(self.arg["""offense-agents"""]))
        if (int(self.arg["defense-npcs"])>0):
            cmd.append("--defense-npcs="+str(self.arg["""defense-npcs"""]))
        if (int(self.arg["offense-npcs"])>0):
            cmd.append("--offense-npcs="+str(self.arg["""offense-npcs"""]))
        if (int(self.arg["defense-agents"])>0):
            cmd.append("--defense-agents="+str(self.arg["""defense-agents"""]))
        #cmd.append("--no-syn")
        print cmd
        self.p=subprocess.Popen(cmd)
        self.pid=self.p.pid
        print "Started HFO Environment"

    def end(self):
        os.kill(self.pid,signal.SIGINT)
        time.sleep(2)
        print "Environment terminated"
