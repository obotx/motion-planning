
import numpy as np
from ompl import base as ob
from ompl import geometric as og


FINGER_DOF = 11
DEFAULT_PLAN_TIMEOUT = 0.5
DEFAULT_PATH_WAYPOINTS = 20

DEFAULT_CTRL_LO = -0.5
DEFAULT_CTRL_HI =  1.0


class _FingerValidityChecker(ob.StateValidityChecker):

    def __init__(self, si):
        super().__init__(si)

    def isValid(self, state):  # noqa: N802 (OMPL API)
        return True


class FingerBridge:

    def __init__(self, model, gripper_act_indices,
                 alpha_min_deg=None, timeout=DEFAULT_PLAN_TIMEOUT):
        if len(gripper_act_indices) < FINGER_DOF:
            raise ValueError(
                f"FingerBridge needs {FINGER_DOF} actuator indices, "
                f"got {len(gripper_act_indices)}")

        self._model    = model
        self._act_ids  = [int(a) for a in gripper_act_indices[:FINGER_DOF]]
        self._timeout  = float(timeout)

        space = ob.RealVectorStateSpace(FINGER_DOF)
        bounds = ob.RealVectorBounds(FINGER_DOF)
        for i, aid in enumerate(self._act_ids):
            cr = model.actuator_ctrlrange[aid]
            lo = float(cr[0])
            hi = float(cr[1])
            if lo == 0.0 and hi == 0.0:
                lo, hi = DEFAULT_CTRL_LO, DEFAULT_CTRL_HI
            if hi - lo < 1e-6:
                hi = lo + 1e-3
            bounds.setLow(i,  lo)
            bounds.setHigh(i, hi)
        space.setBounds(bounds)

        si = ob.SpaceInformation(space)
        si.setStateValidityChecker(_FingerValidityChecker(si))
        si.setStateValidityCheckingResolution(0.05)
        si.setup()

        self._space = space
        self._si    = si

        print(f"[FingerBridge] OMPL state space ready  DOF={FINGER_DOF}  "
              f"act_ids={self._act_ids}")


    def plan(self, q_start, q_goal,
             timeout=None, n_waypoints=DEFAULT_PATH_WAYPOINTS):
        if len(q_start) < FINGER_DOF or len(q_goal) < FINGER_DOF:
            raise ValueError(
                f"FingerBridge.plan needs {FINGER_DOF}-DOF start/goal")

        timeout = float(self._timeout if timeout is None else timeout)

        bounds = self._space.getBounds()
        def _clamp(values):
            return [
                max(float(bounds.low[i]),
                    min(float(bounds.high[i]), float(values[i])))
                for i in range(FINGER_DOF)
            ]
        q_start = _clamp(q_start)
        q_goal  = _clamp(q_goal)

        start_state = self._si.allocState()
        goal_state  = self._si.allocState()
        for i in range(FINGER_DOF):
            start_state[i] = q_start[i]
            goal_state[i]  = q_goal[i]

        pdef = ob.ProblemDefinition(self._si)
        pdef.addStartState(start_state)
        pdef.setGoalState(goal_state)

        planner = og.RRTConnect(self._si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        if not bool(planner.solve(timeout)):
            return None

        path = pdef.getSolutionPath()
        if n_waypoints and n_waypoints > path.getStateCount():
            path.interpolate(n_waypoints)

        waypoints = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            waypoints.append([float(state[j]) for j in range(FINGER_DOF)])

        if len(waypoints) >= 2:
            sign = [
                1.0 if q_goal[j] > q_start[j] else
                (-1.0 if q_goal[j] < q_start[j] else 0.0)
                for j in range(FINGER_DOF)
            ]
            for i in range(1, len(waypoints) - 1):
                for j in range(FINGER_DOF):
                    if sign[j] == 0.0:
                        waypoints[i][j] = q_start[j]
                        continue
                    prev = waypoints[i - 1][j]
                    cur  = waypoints[i][j]
                    if sign[j] > 0.0 and cur < prev:
                        waypoints[i][j] = prev
                    elif sign[j] < 0.0 and cur > prev:
                        waypoints[i][j] = prev
                    if sign[j] > 0.0 and waypoints[i][j] > q_goal[j]:
                        waypoints[i][j] = q_goal[j]
                    elif sign[j] < 0.0 and waypoints[i][j] < q_goal[j]:
                        waypoints[i][j] = q_goal[j]
        return waypoints
