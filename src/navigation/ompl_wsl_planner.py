import sys
import json
import numpy as np
import mujoco
from ompl import base as ob
from ompl import geometric as og

# --- Parse arguments ---
start_x = float(sys.argv[1])
start_y = float(sys.argv[2])
goal_x = float(sys.argv[3])
goal_y = float(sys.argv[4])

# --- Load MuJoCo model ---
model = mujoco.MjModel.from_xml_path(
    "/mnt/c/Users/User1/Downloads/milestone1/base-motion/assets/scene.xml"
)
data = mujoco.MjData(model)

base_x_id = 0
base_y_id = 1

def is_state_valid(state):
    x = state[0]
    y = state[1]

    data.qpos[base_x_id] = x
    data.qpos[base_y_id] = y

    mujoco.mj_forward(model, data)

    return data.ncon == 0

# --- Define state space ---
space = ob.RealVectorStateSpace(2)
bounds = ob.RealVectorBounds(2)
bounds.setLow(-5)
bounds.setHigh(5)
space.setBounds(bounds)

si = ob.SpaceInformation(space)
si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))
si.setup()

start = ob.State(space)
start()[0] = start_x
start()[1] = start_y

goal = ob.State(space)
goal()[0] = goal_x
goal()[1] = goal_y

pdef = ob.ProblemDefinition(si)
pdef.setStartAndGoalStates(start, goal)

planner = og.RRTstar(si)
planner.setProblemDefinition(pdef)
planner.setup()

solved = planner.solve(2.0)

if not solved:
    print(json.dumps([]))
    sys.exit(0)

path = pdef.getSolutionPath()
states = path.getStates()

result = []
for s in states:
    result.append([s[0], s[1]])

print(json.dumps(result))