import sys, json, math
from ompl import base as ob
from ompl import geometric as og

FLOOR_X      = (0.0, 8.0)
FLOOR_Y      = (-8.0, 0.0)
ROBOT_RADIUS = 0.55   # increased — keeps robot well clear of shelf edge

OBSTACLE_RECTS = [
    (2.5,  8.05, -5.2, -3.1),   # shelf — extended slightly top and bottom
    (-0.1, 0.1,  -8.0,  0.0),
    (7.9,  8.1,  -8.0,  0.0),
    (0.0,  8.0,  -0.1,  0.1),
]

class ValidityChecker(ob.StateValidityChecker):
    def __init__(self, si):
        super().__init__(si)
    def isValid(self, state):
        x, y = state[0], state[1]
        if not (FLOOR_X[0] <= x <= FLOOR_X[1] and FLOOR_Y[0] <= y <= FLOOR_Y[1]):
            return False
        r = ROBOT_RADIUS
        for (x0, x1, y0, y1) in OBSTACLE_RECTS:
            if (x0 - r) <= x <= (x1 + r) and (y0 - r) <= y <= (y1 + r):
                return False
        return True

def nudge(si, space, xy, radius=0.8, tries=24):
    state = space.allocState()
    for i in range(tries):
        angle = 2 * math.pi * i / tries
        state[0] = xy[0] + radius * math.cos(angle)
        state[1] = xy[1] + radius * math.sin(angle)
        if si.isValid(state):
            return state
    return None

def plan(start_xy, goal_xy, solve_time=3.0):
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, FLOOR_X[0]);  bounds.setHigh(0, FLOOR_X[1])
    bounds.setLow(1, FLOOR_Y[0]);  bounds.setHigh(1, FLOOR_Y[1])
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ValidityChecker(si))
    si.setStateValidityCheckingResolution(0.01)
    si.setup()

    start = space.allocState()
    start[0], start[1] = float(start_xy[0]), float(start_xy[1])
    goal  = space.allocState()
    goal[0],  goal[1]  = float(goal_xy[0]),  float(goal_xy[1])

    if not si.isValid(start): start = nudge(si, space, start_xy)
    if start is None: return None
    if not si.isValid(goal):  goal  = nudge(si, space, goal_xy)
    if goal is None:  return None

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)

    planner = og.RRTstar(si)
    planner.setProblemDefinition(pdef)
    planner.setRange(0.5)
    planner.setup()

    if not planner.solve(solve_time):
        return None

    path = pdef.getSolutionPath()
    path.interpolate(40)
    waypoints = [[path.getState(i)[0], path.getState(i)[1]]
                 for i in range(path.getStateCount())]
    waypoints[-1] = list(goal_xy)
    return waypoints

if __name__ == "__main__":
    try:
        inp  = json.loads(sys.stdin.read())
        path = plan(inp["start"], inp["goal"], inp.get("solve_time", 3.0))
        print(json.dumps(path if path else {"error": "No path found"}), flush=True)
    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)