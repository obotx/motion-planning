"""
finger_planner.py — OMPL motion planner for the MORPH-I gripper fingers
=========================================================================

OMPL-based planner for the 11-DOF gripper finger joint space.

State space: 11-DOF RealVectorStateSpace
    [c_j1, c_j2, c_j3, b_j1, b_j2, b_j3, a_j1, a_j2, a_j3,
     palm_finger_c_joint_1, palm_finger_b_joint_1]

Validity check: any in-bounds state is valid (no obstacles in finger
workspace; held-object contacts disabled at the executor level).

Planner: RRTConnect, with the solution path interpolated to a
configurable number of waypoints.

Usage (from grasp_executor.GraspExecutor):

    self._finger_bridge = FingerBridge(self.sim.model,
                                        self.sim.gripper_ids_left[:11])

    waypoints = self._finger_bridge.plan(current_ctrl, target_ctrl,
                                          timeout=0.5)
    if waypoints is None:
        ...
    else:
        ...
"""

import numpy as np
from ompl import base as ob
from ompl import geometric as og


# 9 finger joints (3 fingers × 3 joints) + 2 palm spread joints
FINGER_DOF = 11
DEFAULT_PLAN_TIMEOUT = 0.5
DEFAULT_PATH_WAYPOINTS = 20

# Fallback ctrl range when actuator has no configured ctrlrange.
DEFAULT_CTRL_LO = -0.5
DEFAULT_CTRL_HI =  1.0


class _FingerValidityChecker(ob.StateValidityChecker):
    """Trivial validity checker — every in-bounds state is valid."""

    def __init__(self, si):
        super().__init__(si)

    def isValid(self, state):  # noqa: N802 (OMPL API)
        return True


class FingerBridge:
    """
    OMPL motion planner for the gripper's 11 finger DOFs.

    Owns its SpaceInformation and validity checker, runs RRTConnect with
    a configurable timeout, and returns a list of waypoints in
    joint-ctrl space.
    """

    def __init__(self, model, gripper_act_indices,
                 alpha_min_deg=None, timeout=DEFAULT_PLAN_TIMEOUT):
        """
        Args:
            model:                 MuJoCo MjModel (read for actuator ctrl ranges)
            gripper_act_indices:   list of 11 actuator IDs, in order:
                [finger_c_j1, finger_c_j2, finger_c_j3,
                 finger_b_j1, finger_b_j2, finger_b_j3,
                 finger_a_j1, finger_a_j2, finger_a_j3,
                 palm_finger_c_joint_1, palm_finger_b_joint_1]
            alpha_min_deg:         unused (kept for API symmetry)
            timeout:               default plan timeout in seconds
        """
        if len(gripper_act_indices) < FINGER_DOF:
            raise ValueError(
                f"FingerBridge needs {FINGER_DOF} actuator indices, "
                f"got {len(gripper_act_indices)}")

        self._model    = model
        self._act_ids  = [int(a) for a in gripper_act_indices[:FINGER_DOF]]
        self._timeout  = float(timeout)

        # Bounds derived from each actuator's ctrlrange; fall back to
        # DEFAULT_CTRL_LO/HI for actuators with no configured range.
        space = ob.RealVectorStateSpace(FINGER_DOF)
        bounds = ob.RealVectorBounds(FINGER_DOF)
        for i, aid in enumerate(self._act_ids):
            cr = model.actuator_ctrlrange[aid]
            lo = float(cr[0])
            hi = float(cr[1])
            if lo == 0.0 and hi == 0.0:
                lo, hi = DEFAULT_CTRL_LO, DEFAULT_CTRL_HI
            # Avoid degenerate zero-width bounds.
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

    # ── Public API ──────────────────────────────────────────────────────

    def plan(self, q_start, q_goal,
             timeout=None, n_waypoints=DEFAULT_PATH_WAYPOINTS):
        """
        Plan a motion from q_start to q_goal in the 11-DOF finger
        ctrl space.

        Args:
            q_start:        list/array of FINGER_DOF current ctrl values
            q_goal:         list/array of FINGER_DOF target ctrl values
            timeout:        plan timeout in seconds (default: self._timeout)
            n_waypoints:    interpolate the solution path to this many
                            states (more = smoother execution)

        Returns:
            list of n_waypoints lists, each FINGER_DOF floats — the
            planned trajectory in ctrl space; or None on planning failure.
        """
        if len(q_start) < FINGER_DOF or len(q_goal) < FINGER_DOF:
            raise ValueError(
                f"FingerBridge.plan needs {FINGER_DOF}-DOF start/goal")

        timeout = float(self._timeout if timeout is None else timeout)

        # Clamp inputs to configured bounds to avoid OMPL "start state
        # invalid" errors on out-of-range ctrl values.
        bounds = self._space.getBounds()
        def _clamp(values):
            return [
                max(float(bounds.low[i]),
                    min(float(bounds.high[i]), float(values[i])))
                for i in range(FINGER_DOF)
            ]
        q_start = _clamp(q_start)
        q_goal  = _clamp(q_goal)

        # OMPL 2.0: use si.allocState(); ob.State() is not exposed in
        # the new nanobind bindings.
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
        return waypoints
