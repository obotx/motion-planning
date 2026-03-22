from threading import local
import mujoco, os
import glfw
import numpy as np
import datetime, cv2
from scipy.optimize import minimize
import argparse

np.set_printoptions(suppress=True, precision=4)

def wrap_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def quaternion_to_matrix(q):
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    if n < 1e-10:
        raise ValueError("Quaternion has near-zero norm")

    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z

    R = np.array([
        [1.0 - (yy + zz),        xy - wz,        xz + wy],
        [       xy + wz, 1.0 - (xx + zz),        yz - wx],
        [       xz - wy,        yz + wx, 1.0 - (xx + yy)]
    ])
    return R

def quaternion_inverse(q):
    q_conj = quaternion_conjugate(q)
    norm_sq = np.dot(q, q)
    return q_conj / norm_sq

def rotate_quaternion(quat, axis, angle):
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    cos_half = np.cos(angle_rad / 2)
    sin_half = np.sin(angle_rad / 2)
    delta_quat = np.array([cos_half, sin_half * axis[0], sin_half * axis[1], sin_half * axis[2]])
    new_quat = quaternion_multiply(quat, delta_quat)
    new_quat /= np.linalg.norm(new_quat)
    return new_quat

def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_rotate_vector(quat, vec):
    vec_quat = np.array([0.0, vec[0], vec[1], vec[2]])
    qv = quaternion_multiply(quat, vec_quat)
    rotated_quat = quaternion_multiply(qv, quaternion_conjugate(quat))
    return rotated_quat[1:]

class ParallelRobot:
    GRIPPER_OPEN_POS = 0.0
    GRIPPER_CLOSED_POS = 255.0
    GRIPPER_INCREMENT = 20.0
    
    GRIPPER_SIDE_NORMAL = 0.0
    GRIPPER_SIDE_OPEN = 30.0
    GRIPPER_SIDE_CLOSE = -30.0
    GRIPPER_INC_SIDE = 10.0
    
    GRIPPER_TIP_OPEN = -255.0
    GRIPPER_TIP_CLOSE = 255.0
    
    SWITCH_DISTANCE = 0.3   
    INC_STEP = 0.5
    DAMPING = 8e-4
    DT = 0.002
        
    base_integral=0 
    base_prev_error=0

    r = 0.1  
    D = 0.55   

    mobile_dot = np.zeros(4)
    target_vel = np.zeros(4)
    command = np.zeros(4)

    integral_x = 0.0
    integral_y = 0.0
    integral_yaw = 0.0
    prev_delta_x = 0.0
    prev_delta_y = 0.0
    prev_delta_yaw = 0.0
    
    JOINT_NAMES = [
        "BaseJoint",
        "ColumnLeftBearingJoint",
        "ColumnRightBearingJoint",
        "ArmLeftJoint",
        "ArmRightJoint",
    ]
    ACTUATOR_NAMES = [
        "BaseJointMotor",
        "ColumnLeftBearingJointMotor",
        "ColumnRightBearingJointMotor",
        "ArmLeftJointMotor",
        "ArmRightJointMotor",
    ]
    GRIPPER_ACT=[
        "finger_c_joint_1",
        "finger_b_joint_1",
        "finger_a_joint_1",
        
        "finger_c_joint_3",
        "finger_b_joint_3",
        "finger_a_joint_3",        
        
        "palm_finger_c_joint",
        "palm_finger_b_joint"
    ]
    

    def __init__(self, path: str, run_mode: str, record: bool):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self._initialize_ids()
        self._initialize_arrays()
        
        self._terminate = False  
        self.paused = False 
        self.run_mode = run_mode.lower()
        self.record = record

        if self.run_mode == "glfw":
            if not glfw.init():
                raise RuntimeError("GLFW failed to initialize")
            self.window = glfw.create_window(1200, 900, "Gripper Simulation", None, None)
            if not self.window:
                glfw.terminate()
                raise RuntimeError("GLFW failed to create window")
            glfw.make_context_current(self.window)

            glfw.set_key_callback(self.window, self.on_key)
            glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
            glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)

            self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self.viewport = mujoco.MjrRect(0, 0, 1200, 900)
            self.camera = mujoco.MjvCamera()
            self.scene = mujoco.MjvScene(self.model, maxgeom=500)
            # self.scene.flags = mujoco.mjtRndFlag.mjRND_SHADOW #disable shadow

            self.opt = mujoco.MjvOption()
        
            self._last_mouse_x = 0
            self._last_mouse_y = 0
            self._mouse_left_pressed = False
            self._mouse_right_pressed = False
            self._mouse_middle_pressed = False
            
        if self.run_mode == "cv":
            self.top_camera_name = "top_view"
            self.pov_camera_name = "pov"

            self.top_camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.top_camera_name
            )
            self.pov_camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.pov_camera_name
            )

            self.renderer_top = mujoco.Renderer(self.model, height=640, width=1024)
            self.renderer_pov = mujoco.Renderer(self.model, height=640, width=1024)

            self.model.vis.global_.offheight = 640
            self.model.vis.global_.offwidth = 1024

        self.current_waypoint_idx = 0
        self.time_at_current_waypoint = 0.0
        self.speed = 3  
        self.angular_speed = 3  
        self.grab_time = 0.0
        self.grab_hold_duration = 0.5
        self.grab_move_speed = 0.2
        self.progress = 0
        
        self.print_controls()
        
        base_pos = self.data.mocap_pos[self.mocap_base_id].copy()
        base_quat = self.data.mocap_quat[self.mocap_base_id].copy()
        arm_pos = self.data.mocap_pos[self.mocap_arm_id].copy()
        arm_quat = self.data.mocap_quat[self.mocap_arm_id].copy()
        self.arm_rel_pos = quaternion_rotate_vector(quaternion_conjugate(base_quat), arm_pos - base_pos)
        self.arm_rel_quat = quaternion_multiply(quaternion_conjugate(base_quat), arm_quat)
        
        self.h1, self.h2, self.l1, self.l2 = 0, 0, 0, 0
        self.top_video_writer = None
        self.pov_video_writer = None
        
    def update_arm_from_base(self):
        base_pos = self.data.xpos[self.base_id].copy()
        base_quat = self.data.xquat[self.base_id].copy()
        norm = np.linalg.norm(base_quat)
        base_quat /= norm

        correction = np.array([0.0, -1.0, 0.0, 0.0])
        base_quat = quaternion_multiply(base_quat, correction)
        base_quat /= np.linalg.norm(base_quat)

        rel_pos = self.arm_rel_pos
        self.data.mocap_pos[self.mocap_arm_id] = (
            base_pos + quaternion_rotate_vector(base_quat, rel_pos)
        )

        rel_quat = self.arm_rel_quat
        new_quat = quaternion_multiply(base_quat, rel_quat)
        self.data.mocap_quat[self.mocap_arm_id] = new_quat / np.linalg.norm(new_quat)

    def _initialize_ids(self):
        self.site_id = self.model.site("attachment_site").id
        self.arm_link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Arm")
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Link1")
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_footprint")

        self.dof_ids = np.array([self.model.joint(name).id for name in self.JOINT_NAMES])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in self.ACTUATOR_NAMES])
        self.gripper_act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.GRIPPER_ACT]
        self.mocap_arm_id = self.model.body("target_arm").mocapid[0]
        self.mocap_base_id = self.model.body("target_base").mocapid[0]

        self.target_arm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_arm")
        self.target_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_base")

        self.qpos_indices = []
        self.qvel_indices = []

        for name in self.JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.model.jnt_qposadr[joint_id]
            dof_adr = self.model.jnt_dofadr[joint_id]
            joint_type = self.model.jnt_type[joint_id]

            if joint_type in [3, 2]:
                self.qpos_indices.append(qpos_adr)
                self.qvel_indices.append(dof_adr)
            elif joint_type == 1:
                self.qpos_indices.extend(range(qpos_adr, qpos_adr + 3))
                self.qvel_indices.extend(range(dof_adr, dof_adr + 3))
            elif joint_type == 0:
                self.qpos_indices.extend(range(qpos_adr, qpos_adr + 6))
                self.qvel_indices.extend(range(dof_adr, dof_adr + 6))

        self.qpos_indices = np.array(self.qpos_indices)
        self.qvel_indices = np.array(self.qvel_indices)

    def _initialize_arrays(self):
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.error = np.zeros(6)
        self.error_pos = np.zeros(3)
        self.error_ori = np.zeros(3)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        
    def configure_model(self):
        self.model.opt.timestep = self.DT
        self.model.body_gravcomp[:] = True
        
    def control_base(self, target_arm_id, alpha=0):
        k_p = 5.0
        k_i = 0.1
        k_d = 0.8
        
        k_p_theta = 5.0
        k_i_theta = 0.1
        k_d_theta = 0.8

        self.mobile_dot[0] = self.data.qvel[19]
        self.mobile_dot[1] = self.data.qvel[6] 
        self.mobile_dot[2] = self.data.qvel[45]
        self.mobile_dot[3] = self.data.qvel[32]

        target_x, target_y = self.data.xpos[target_arm_id, 0], self.data.xpos[target_arm_id, 1]
        w, x, y, z = self.data.xquat[target_arm_id]
        target_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))

        current_x, current_y = self.data.xpos[self.base_id, 0], self.data.xpos[self.base_id, 1]
        w, x, y, z = self.data.xquat[self.base_id]
        current_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))

        delta_x = target_x - current_x
        delta_y = target_y - current_y
        delta_yaw = np.arctan2(np.sin(target_yaw - current_yaw),
                            np.cos(target_yaw - current_yaw))

        delta_x_local = np.cos(current_yaw) * delta_x + np.sin(current_yaw) * delta_y
        delta_y_local = -np.sin(current_yaw) * delta_x + np.cos(current_yaw) * delta_y

        self.integral_x += delta_x_local * self.model.opt.timestep
        self.integral_y += delta_y_local * self.model.opt.timestep
        self.integral_yaw += delta_yaw * self.model.opt.timestep
        deriv_x_local = (delta_x_local - self.prev_delta_x) / self.model.opt.timestep
        deriv_y_local = (delta_y_local - self.prev_delta_y) / self.model.opt.timestep
        deriv_yaw = (delta_yaw - self.prev_delta_yaw) / self.model.opt.timestep

        self.deriv_x = alpha * getattr(self, "deriv_x", 0.0) + (1 - alpha) * deriv_x_local
        self.deriv_y = alpha * getattr(self, "deriv_y", 0.0) + (1 - alpha) * deriv_y_local
        self.deriv_yaw = alpha * getattr(self, "deriv_yaw", 0.0) + (1 - alpha) * deriv_yaw

        self.prev_delta_x = delta_x_local
        self.prev_delta_y = delta_y_local
        self.prev_delta_yaw = delta_yaw

        v_x_local = k_p * delta_x_local + k_i * self.integral_x + k_d * self.deriv_x
        v_y_local = k_p * delta_y_local + k_i * self.integral_y + k_d * self.deriv_y
        omega = k_p_theta * delta_yaw + k_i_theta * self.integral_yaw + k_d_theta * self.deriv_yaw

        self.target_vel[0] = (v_x_local - v_y_local - omega * self.D) / self.r
        self.target_vel[1] = (v_x_local + v_y_local + omega * self.D) / self.r
        self.target_vel[2] = (v_x_local + v_y_local - omega * self.D) / self.r
        self.target_vel[3] = (v_x_local - v_y_local + omega * self.D) / self.r

        self.command = self.target_vel - self.mobile_dot
        eps = 0.05
        self.command[np.abs(self.command) < eps] = 0.0

        self.data.ctrl[0] = self.command[1]
        self.data.ctrl[1] = self.command[0]
        self.data.ctrl[2] = self.command[3]
        self.data.ctrl[3] = self.command[2]
        
    def ik_solution(
        self, err=None, local_target=None,
        D2=0.2,
        l_bounds=np.array([[0.4, 1.0],
                           [0, 0.6]]),
        h_bounds=np.array([[0.0, 1.1],
                           [0.0, 1.1]]),
        theta_bounds=(-3.14, 3.14),
        recompute_thresh=1e-3
    ):
        arm_base_pos = self.data.xpos[self.arm_link_id]
        base_quat = self.data.xquat[self.base_id] / np.linalg.norm(self.data.xquat[self.base_id])
        
        R_base = quaternion_to_matrix(base_quat) 
        err_base = err
        
        ee_world = self.data.xpos[self.end_effector_id]
        ee_base  = R_base.T @ (ee_world - arm_base_pos)

        target = local_target if local_target is not None else ee_base + err_base
        
        if hasattr(self, "_last_ik_target"):
            if np.linalg.norm(target - np.array(self._last_ik_target)) < recompute_thresh:
                return
        self._last_ik_target = target.copy()
        
        def normalize_angle(theta):
            return (theta + np.pi) % (2*np.pi) - np.pi

        def bases_from_x(x):
            h1, h2, th = x
            th = normalize_angle(th)
            c, s = np.cos(th), np.sin(th)
            b1 = np.array([-D2/2 * c, -D2/2 * s, h1])
            b2 = np.array([ D2/2 * c,  D2/2 * s, h2])
            return b1, b2

        def obj(x):
            b1, b2 = bases_from_x(x)
            v1, v2 = target - b1, target - b2
            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            denom = (l1 * l2) if l1*l2 != 0 else 1e-12
            cos_th = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
            return -np.arccos(cos_th)
        
        def l1_min_c(x):
            b1, _ = bases_from_x(x)
            return np.linalg.norm(target - b1)**2 - l_bounds[0,0]**2
        def l1_max_c(x):
            b1, _ = bases_from_x(x)
            return l_bounds[0,1]**2 - np.linalg.norm(target - b1)**2
        def l2_min_c(x):
            _, b2 = bases_from_x(x)
            return np.linalg.norm(target - b2)**2 - l_bounds[1,0]**2
        def l2_max_c(x):
            _, b2 = bases_from_x(x)
            return l_bounds[1,1]**2 - np.linalg.norm(target - b2)**2
        def angle_min_c(x, min_deg=20.0):
            b1, b2 = bases_from_x(x)
            v1, v2 = target - b1, target - b2
            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            denom = (l1 * l2) if l1*l2 != 0 else 1e-12
            cos_th = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
            angle = np.arccos(cos_th) 
            return angle - np.deg2rad(min_deg)
        tol = 1e-3 
        def proj_parallel_pos(x, t=tol):
            return  t - (float(target[0]) * (-np.sin(x[2])) + float(target[1]) * np.cos(x[2]))
        def proj_parallel_neg(x, t=tol):
            return  t + (float(target[0]) * (-np.sin(x[2])) + float(target[1]) * np.cos(x[2]))
        
        z_threshold = 0.05
        arm_length  = 1.15
        def l1_end_z_c(x):
            b1, _ = bases_from_x(x)
            end_eff = target
            l1 = np.linalg.norm(end_eff - b1)
            if l1 < l_bounds[0,1]:
                dir1 = (end_eff - b1) / max(np.linalg.norm(end_eff - b1), 1e-12)
                l1_end = b1 - dir1 * (arm_length - l1)
                return l1_end[2] - (-z_threshold)
            return 1.0

        def l2_end_z_c(x):
            _, b2 = bases_from_x(x)
            end_eff = target
            l2 = np.linalg.norm(end_eff - b2)
            if l2 < l_bounds[1,1]:
                dir2 = (end_eff - b2) / max(np.linalg.norm(end_eff - b2), 1e-12)
                l2_end = b2 - dir2 * (arm_length - l2)
                return l2_end[2] - (-z_threshold)
            return 1.0

        constraints = [
            {'type': 'ineq', 'fun': angle_min_c},
            {'type': 'ineq', 'fun': l1_min_c},
            {'type': 'ineq', 'fun': l1_max_c},
            {'type': 'ineq', 'fun': l2_min_c},
            {'type': 'ineq', 'fun': l2_max_c},
            {'type': 'ineq', 'fun': proj_parallel_pos},
            {'type': 'ineq', 'fun': proj_parallel_neg},
            {'type': 'ineq', 'fun': l1_end_z_c},
            {'type': 'ineq', 'fun': l2_end_z_c},
        ]

        bounds = [
            (float(h_bounds[0,0]), float(h_bounds[0,1])),
            (float(h_bounds[1,0]), float(h_bounds[1,1])),
            (float(theta_bounds[0]), float(theta_bounds[1]))
        ]

        x0 = np.array([
            self.data.qpos[self.qpos_indices[2]],
            self.data.qpos[self.qpos_indices[1]],
            self.data.qpos[self.qpos_indices[0]]
        ])
        theta_cur = self.data.qpos[self.qpos_indices[0]]
        theta_guess = (theta_cur + np.pi) % (2*np.pi) - np.pi
        x0[2] = theta_guess
        
        x0 = np.clip(
            x0,
            [b[0] for b in bounds],
            [b[1] for b in bounds]
        )

        res = minimize(
            obj, x0,
            method='SLSQP',
            tol=1e-5,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-5, 'maxiter': 15, 'eps':1e-8, 'disp': False}
        )
      
        h1, h2, theta = res.x
        self.h1, self.h2, self.theta = float(h1), float(h2), float(theta)
        self.success = res.success
        if not res.success:
          return

        b1, b2 = bases_from_x([self.h1, self.h2, self.theta])
        v1 = target - b1
        v2 = target - b2
        self.l1 = float(np.linalg.norm(v1))
        self.l2 = float(np.linalg.norm(v2))

        denom = (self.l1 * self.l2) if self.l1*self.l2 != 0 else 1e-12
        cos_th = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        self.best_angle = float(np.degrees(np.arccos(cos_th)))
            
    def control_arm(self, target_arm_id):
        alpha = 0.01 
        act_offset = np.array([0, 0, -0.075428, -0.303811])
        arm_base_pos  = self.data.xpos[self.arm_link_id]
        base_quat = self.data.xquat[self.base_id] / np.linalg.norm(self.data.xquat[self.base_id])
        R_base    = quaternion_to_matrix(base_quat)

        target_world = self.data.mocap_pos[target_arm_id] - np.array([0, 0, 0.05])
        ee_world     = self.data.xpos[self.end_effector_id]

        target_base = R_base.T @ (target_world - arm_base_pos)
        ee_base     = R_base.T @ (ee_world - arm_base_pos)
        
        error = target_base - ee_base
        self.ik_solution(err=error)
        
        solution=np.array([self.h2, self.h1, self.l2, self.l1]) + act_offset
        act = solution*100 
        act = (1 - alpha) * self.data.ctrl[self.actuator_ids[1:]] + alpha * act
        lo = self.model.actuator_ctrlrange[self.actuator_ids[1:], 0]
        hi = self.model.actuator_ctrlrange[self.actuator_ids[1:], 1]
        act_clipped = np.clip(act, lo, hi)

        self.control_base_arm(self.theta)
        if self.success :
            self.data.ctrl[self.actuator_ids[1:]] = act_clipped
            
    def control_base_arm(self, target_angle, kp=5, ki=0.05, kd=2):
        joint_id = self.model.joint("BaseJoint").id
        qpos_adr = self.model.jnt_qposadr[joint_id]
        qvel_adr = self.model.jnt_dofadr[joint_id]

        qpos = self.data.qpos[qpos_adr]
        qvel = self.data.qvel[qvel_adr]

        error = (target_angle - qpos + np.pi) % (2*np.pi) - np.pi
        self.base_integral += error * self.model.opt.timestep

        derivative = (error - self.base_prev_error) / self.model.opt.timestep
        self.base_prev_error = error

        torque = kp * error + ki * self.base_integral + kd * derivative

        lo, hi = self.model.actuator_ctrlrange[self.actuator_ids[0]]
        self.data.ctrl[self.actuator_ids[0]] = np.clip(torque, lo, hi)

    def step_simulation(self):
        self.update_arm_from_base()
        self.control_arm(self.mocap_arm_id)
        self.control_base(self.target_base_id)
        mujoco.mj_step(self.model, self.data, nstep=5)
        if self.run_mode == "cv":
            self.camera_display()
        elif self.run_mode == "glfw":
            mujoco.mjv_updateScene(
                self.model, self.data, self.opt, None, self.camera, 0xFF, self.scene
            )
            mujoco.mjr_render(self.viewport, self.scene, self.ctx)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        
    def camera_display(self):
        self.frame_count = getattr(self, 'frame_count', 0)
        
        self.renderer_top.update_scene(self.data, camera=self.top_camera_id)
        rgb_top = self.renderer_top.render()
        if rgb_top is None or rgb_top.size == 0:
            print("Error: Top view rendering failed")
            return
        bgr_top = cv2.cvtColor(rgb_top, cv2.COLOR_RGB2BGR)
        
        self.renderer_pov.update_scene(self.data, camera=self.pov_camera_id)
        rgb_side = self.renderer_pov.render()
        if rgb_side is None or rgb_side.size == 0:
            print("Error: Side view rendering failed")
            return
        bgr_side = cv2.cvtColor(rgb_side, cv2.COLOR_RGB2BGR)

        if self.top_video_writer is not None and self.top_video_writer.isOpened():
            self.top_video_writer.write(bgr_top)
        if self.pov_video_writer is not None and self.pov_video_writer.isOpened():
            self.pov_video_writer.write(bgr_side)
        self.frame_count += 1

        cv2.imshow("MuJoCo Top View", bgr_top)
        cv2.imshow("MuJoCo Side View", bgr_side)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self._terminate = True

    def run_cv(self):
        if self.record:
            output_dir = "output_videos"
            os.makedirs(output_dir, exist_ok=True)
            
            frame_width, frame_height = self.model.vis.global_.offwidth, self.model.vis.global_.offheight 
            fps = 30
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            top_video_file = os.path.join(output_dir, f"top_view_{timestamp}.mp4")
            pov_video_file = os.path.join(output_dir, f"pov_view_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG') 

            self.top_video_writer = cv2.VideoWriter(top_video_file, fourcc, fps, (frame_width, frame_height))
            self.pov_video_writer = cv2.VideoWriter(pov_video_file, fourcc, fps, (frame_width, frame_height))

            if not self.top_video_writer.isOpened():
                print(f"Error: Failed to open top video writer for {top_video_file}")
            if not self.pov_video_writer.isOpened():
                print(f"Error: Failed to open side video writer for {pov_video_file}")

        try:
            mujoco.mj_step(self.model, self.data, nstep=1)
            self._terminate = False
            while not self._terminate:
                self.step_simulation()
        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            if self.record:
                if self.top_video_writer is not None:
                    self.top_video_writer.release()
                    print(f"Released top video writer: {top_video_file}")
                if self.pov_video_writer is not None:
                    self.pov_video_writer.release()
                    print(f"Released side video writer: {pov_video_file}")
            cv2.destroyAllWindows()
            self.renderer_top.close()
            self.renderer_pov.close()

    def run_glfw(self):
        mujoco.mj_step(self.model, self.data, nstep=1)
        while not glfw.window_should_close(self.window) and not self._terminate:
            self.step_simulation()
        glfw.terminate()

    def print_controls(self):
        lines = [
            "MuJoCo Controller - Key Bindings",
            "", 
            "[ General ]",
            "  ESC      : Exit simulation",
            "  ENTER    : Reset simulation",
            "",
            "[ Gripper Control ]",
            "  Z        : Close gripper (jaw)",
            "  X        : Open gripper (jaw)",
            "  C        : Side gripper outward",
            "  V        : Side gripper inward",
            "  B        : Tip gripper open",
            "  N        : Tip gripper close",
            "",
            "[ Arm Movement ]",
            "  W / S    : Forward / Backward",
            "  A / D    : Left / Right",
            "  Q / E    : Up / Down",
            "",
            "[ Base Movement (relative to orientation) ]",
            "  ↑ / ↓    : Forward / Backward",
            "  → / ←    : Right / Left",
            "  T / R    : Rotate CCW / CW",
            "",
            "[ Camera Control (Free Mode) ]",
            "  Mouse Left Drag    : Rotate camera",
            "  Mouse Right Drag   : Pan camera",
            "  Mouse Middle Drag  : Zoom",
            "  Mouse Scroll       : Zoom",
        ]

        width = max(len(line) for line in lines)
        border = "─" * (width + 4)
        lines[1] = "─" * (width)

        print("┌" + border + "┐")
        for line in lines:
            print(f"│  {line.ljust(width)}  │")
        print("└" + border + "┘")
            
    def on_key(self, window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        current_quat = self.data.mocap_quat[0]
        move_amount = 0.01

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.window, True)
            return
        
        if key == glfw.KEY_ENTER:
            mujoco.mj_resetData(self.model, self.data)
            self.gripper_ctrl = 0.0
            return

        if key in (glfw.KEY_Z, glfw.KEY_X):
            for act_id in self.gripper_act_ids[0:3]: 
                current_val = self.data.ctrl[act_id]

                if key == glfw.KEY_Z:
                    new_val = np.clip(
                        current_val + self.GRIPPER_INCREMENT,
                        self.GRIPPER_OPEN_POS,
                        self.GRIPPER_CLOSED_POS
                    )
                elif key == glfw.KEY_X:
                    new_val = np.clip(
                        current_val - self.GRIPPER_INCREMENT,
                        self.GRIPPER_OPEN_POS,
                        self.GRIPPER_CLOSED_POS
                    )

                self.data.ctrl[act_id] = new_val

        if key in (glfw.KEY_C, glfw.KEY_V):
            left_act, right_act = self.gripper_act_ids[-2], self.gripper_act_ids[-1]

            left_val = self.data.ctrl[left_act]
            right_val = self.data.ctrl[right_act]

            if key == glfw.KEY_C:
                new_left = np.clip(left_val + self.GRIPPER_INC_SIDE,
                                self.GRIPPER_SIDE_CLOSE, self.GRIPPER_SIDE_OPEN)
                new_right = np.clip(right_val - self.GRIPPER_INC_SIDE,
                                    self.GRIPPER_SIDE_CLOSE, self.GRIPPER_SIDE_OPEN)

            elif key == glfw.KEY_V:
                new_left = np.clip(left_val - self.GRIPPER_INC_SIDE,
                                self.GRIPPER_SIDE_CLOSE, self.GRIPPER_SIDE_OPEN)
                new_right = np.clip(right_val + self.GRIPPER_INC_SIDE,
                                    self.GRIPPER_SIDE_CLOSE, self.GRIPPER_SIDE_OPEN)

            self.data.ctrl[left_act] = new_left
            self.data.ctrl[right_act] = new_right
            
        if key in (glfw.KEY_B, glfw.KEY_N):
            for act_id in self.gripper_act_ids[3:6]: 
                current_val = self.data.ctrl[act_id]

                if key == glfw.KEY_B:
                    new_val = np.clip(
                        current_val + self.GRIPPER_INCREMENT,
                        self.GRIPPER_TIP_OPEN,
                        self.GRIPPER_TIP_CLOSE
                    )
                elif key == glfw.KEY_N:
                    new_val = np.clip(
                        current_val - self.GRIPPER_INCREMENT,
                        self.GRIPPER_TIP_OPEN,
                        self.GRIPPER_TIP_CLOSE
                    )

                self.data.ctrl[act_id] = new_val
                
        arm_local_direction = np.zeros(3)

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            arm_local_direction += np.array([1.0, 0.0, 0.0])
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            arm_local_direction += np.array([-1.0, 0.0, 0.0])

        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            arm_local_direction += np.array([0.0, 1.0, 0.0])
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            arm_local_direction += np.array([0.0, -1.0, 0.0])

        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
            arm_local_direction += np.array([0.0, 0.0, 1.0])
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
            arm_local_direction += np.array([0.0, 0.0, -1.0])

        if np.linalg.norm(arm_local_direction) > 1e-8:
            arm_local_direction /= np.linalg.norm(arm_local_direction)
            self.arm_rel_pos += arm_local_direction * move_amount

        base_local_direction = np.zeros(3)

        if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
            base_local_direction += np.array([1.0, 0.0, 0.0])
        if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
            base_local_direction += np.array([-1.0, 0.0, 0.0])
        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            base_local_direction += np.array([0.0, 1.0, 0.0])
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            base_local_direction += np.array([0.0, -1.0, 0.0])

        if np.linalg.norm(base_local_direction) > 1e-8:
            base_local_direction /= np.linalg.norm(base_local_direction)
            global_direction = quaternion_rotate_vector(current_quat, base_local_direction)
            self.data.mocap_pos[self.mocap_base_id] += global_direction * move_amount
        
        elif key == glfw.KEY_T:
            self.data.mocap_quat[self.mocap_base_id] = rotate_quaternion(
                self.data.mocap_quat[self.mocap_base_id], np.array([0,0,1]), 1
            )
        elif key == glfw.KEY_R:
            self.data.mocap_quat[self.mocap_base_id] = rotate_quaternion(
                self.data.mocap_quat[self.mocap_base_id], np.array([0,0,1]), -1
            )

    def _cursor_pos_callback(self, window, xpos, ypos):
        if self.camera.type != mujoco.mjtCamera.mjCAMERA_FREE:
            self._last_mouse_x, self._last_mouse_y = xpos, ypos
            return

        dx = xpos - self._last_mouse_x
        dy = ypos - self._last_mouse_y
        self._last_mouse_x, self._last_mouse_y = xpos, ypos
        factor = 0.001
        if self._mouse_left_pressed:
            mujoco.mjv_moveCamera(
                self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                dx*factor, dy*factor, self.scene, self.camera
            )
        elif self._mouse_right_pressed:
            mujoco.mjv_moveCamera(
                self.model, mujoco.mjtMouse.mjMOUSE_MOVE_H,
                dx*factor, dy*factor, self.scene, self.camera
            )
        elif self._mouse_middle_pressed:
            mujoco.mjv_moveCamera(
                self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                dx*factor, dy*factor, self.scene, self.camera
            )

    def _mouse_button_callback(self, window, button, action, mods):
        if self.camera.type != mujoco.mjtCamera.mjCAMERA_FREE:
            return
        pressed = (action == glfw.PRESS)
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._mouse_left_pressed = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._mouse_right_pressed = pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._mouse_middle_pressed = pressed

    def _scroll_callback(self, window, xoffset, yoffset):
        if self.camera.type != mujoco.mjtCamera.mjCAMERA_FREE:
            return
        factor = 0.05
        mujoco.mjv_moveCamera(
            self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
            0, yoffset*factor, self.scene, self.camera
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MuJoCo Parallel Robot Simulation")
    parser.add_argument("--run", choices=["glfw", "cv"], default="glfw",
                        help="Run mode: 'glfw' or 'cv'")
    parser.add_argument("--record", action="store_true",
                        help="Record video output to MP4 (only applicable with --run cv)")
    args = parser.parse_args()

    if args.record and args.run != "cv":
        print("Warning: --record is only applicable with --run cv. Ignoring --record.")
        args.record = False
        
    xml_path = os.path.join(os.path.dirname(__file__), '..', 'env', 'kitchen_world.xml')
    xml_path = os.path.abspath(xml_path)
    sim = ParallelRobot(xml_path, args.run, args.record)
    if args.run == "glfw":
        sim.run_glfw()
    else:  # cv
        sim.run_cv()