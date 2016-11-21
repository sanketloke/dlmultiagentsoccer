# Actions
# DASH,       // [Low-Level] Dash(power [0,100], direction [-180,180])
# TURN,       // [Low-Level] Turn(direction [-180,180])
# TACKLE,     // [Low-Level] Tackle(direction [-180,180])
# KICK,       // [Low-Level] Kick(power [0,100], direction [-180,180])
# KICK_TO,    // [Mid-Level] Kick_To(target_x [-1,1], target_y [-1,1], speed [0,3])
# MOVE_TO,    // [Mid-Level] Move(target_x [-1,1], target_y [-1,1])
# DRIBBLE_TO, // [Mid-Level] Dribble(target_x [-1,1], target_y [-1,1])
# INTERCEPT,  // [Mid-Level] Intercept(): Intercept the ball
# MOVE,       // [High-Level] Move(): Reposition player according to strategy
# SHOOT,      // [High-Level] Shoot(): Shoot the ball
# PASS,       // [High-Level] Pass(teammate_unum [0,11]): Pass to the most open teammate
# DRIBBLE,    // [High-Level] Dribble(): Offensive dribble
# CATCH,      // [High-Level] Catch(): Catch the ball (Goalie only!)
class Agent(object):


    """docstring for Agent."""
    def __init__(self):
        super(Agent, self).__init__()

    """docstring for perceive"""
    def perceive(self):
        print 'Base Function for perceive'
