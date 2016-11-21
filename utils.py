def statusToString(status):
    if status==0:
        return 'IN_GAME'
    elif status==1:
        return 'GOAL'
    elif status==2:
        return 'CAPTURED_BY_DEFENSE'
    elif status==3:
        return 'OUT_OF_BOUNDS'
    elif status==4:
        return 'OUT_OF_TIME'
    else:
        return 'SERVER_DOWN'
