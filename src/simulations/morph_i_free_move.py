import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco, os
import glfw, time
import numpy as np
import datetime, cv2
from scipy.optimize import minimize
import argparse
from modules.pubsub import IPCPubSub
import threading

np.set_printoptions(suppress=True, precision=4)

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
    DAMPING = 8e-4
    DT = 0.002
        
    base_integral_1=0 
    base_prev_error_1=0
    base_integral_2=0 
    base_prev_error_2=0

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
        "ColumnLeftBearingJoint_1",
        "ColumnRightBearingJoint_1",
        "ArmLeftJoint_1",
        "BaseJoint_1",
        "ColumnLeftBearingJoint_2",
        "ColumnRightBearingJoint_2",
        "ArmLeftJoint_2",
        "BaseJoint_2",
    ]
    ACTUATOR_NAMES = [
        "ColumnLeftBearingJointMotor_1",
        "ColumnRightBearingJointMotor_1",
        "ArmLeftJointMotor_1",
        "BaseJointMotor_1",
        "ColumnLeftBearingJointMotor_2",
        "ColumnRightBearingJointMotor_2",
        "ArmLeftJointMotor_2",
        "BaseJointMotor_2",
    ]
    
    GRIPPER_ACT_LEFT = [
        "finger_c_joint_1_1",
        "finger_c_joint_2_1",
        "finger_c_joint_3_1",
        "finger_b_joint_1_1",
        "finger_b_joint_2_1",
        "finger_b_joint_3_1",
        "finger_a_joint_1_1",
        "finger_a_joint_2_1",
        "finger_a_joint_3_1",
        "palm_finger_c_joint_1",
        "palm_finger_b_joint_1",
        "wrist_X_1",
        "wrist_Y_1",
        "wrist_Z_1",
        "HandBearing_1"
    ]
    
    GRIPPER_ACT_RIGHT = [
        "finger_c_joint_1_2",
        "finger_c_joint_2_2",
        "finger_c_joint_3_2",
        "finger_b_joint_1_2",
        "finger_b_joint_2_2",
        "finger_b_joint_3_2",
        "finger_a_joint_1_2",
        "finger_a_joint_2_2",
        "finger_a_joint_3_2",
        "palm_finger_c_joint_2",
        "palm_finger_b_joint_2",
        "wrist_X_2",
        "wrist_Y_2",
        "wrist_Z_2",
        "HandBearing_2"
    ]
        
    def __init__(self, path: str, run_mode: str, record: bool):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.reset("home")
        self._target_lock = threading.Lock()
        self._initialize_ids()
        self._initialize_arrays()
        
        self._terminate = False  
        self.paused = False 
        self.run_mode = run_mode.lower()
        self.record = record
        self.current_ctrl = np.zeros(len(self.ACTUATOR_NAMES))
        
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 5.0         
        self.camera.azimuth = 90            
        self.camera.elevation = -45         
        self.camera.lookat[:] = [0, 0, 0]
        
        self.use_ik = False 
        self.direct_arm_commands = np.concatenate([self.data.ctrl[self.actuator_ids[0:3]]/100, 
                                                   [0], 
                                                   self.data.ctrl[self.actuator_ids[4:7]]/100, 
                                                   [0]])

        if self.run_mode == "glfw":
            if not glfw.init():
                raise RuntimeError("GLFW failed to initialize")
            self.window = glfw.create_window(1200, 900, "Gripper Simulation", None, None)
            if not self.window:
                glfw.terminate()
                raise RuntimeError("GLFW failed to create window")
            glfw.make_context_current(self.window)
            self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self.viewport = mujoco.MjrRect(0, 0, 1200, 900)
            self.scene = mujoco.MjvScene(self.model, maxgeom=500)
            self.opt = mujoco.MjvOption()

            glfw.set_key_callback(self.window, self.on_key)
            # glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
            # glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)

            # self.scene.flags = mujoco.mjtRndFlag.mjRND_SHADOW #disable shadow
        
            self._last_mouse_x = 0
            self._last_mouse_y = 0
            self._mouse_left_pressed = False
            self._mouse_right_pressed = False
            self._mouse_middle_pressed = False
            
        if self.run_mode == "cv":
            self.renderer_top = mujoco.Renderer(self.model, height=640, width=1024)

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
        
        self.h1, self.h2, self.l1, self.l2 = 0, 0, 0, 0
        self.top_video_writer = None
        self.pov_video_writer = None

        self.ipc = IPCPubSub()
        self.subscriber = self.ipc.create_subscriber()
        self.subscriber.subscribe("target_base", self._on_target_base)
        self.subscriber.subscribe("target_left", self._on_target_left)
        self.subscriber.subscribe("target_right", self._on_target_right)
        self.subscriber.subscribe("ik_mode", self._on_ik_mode)
        self.subscriber.subscribe("u_control", self._on_arm_control)
        self.subscriber.start()
        
    def _initialize_ids(self):
        self.arm_link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Arm")
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Link1")
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_footprint")

        self.dof_ids = np.array([self.model.joint(name).id for name in self.JOINT_NAMES])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in self.ACTUATOR_NAMES])
        self.gripper_ids_left = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.GRIPPER_ACT_LEFT]
        self.gripper_ids_right = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.GRIPPER_ACT_RIGHT]
        
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

        l0, r0 = self.get_encoder()
        self.target_left = np.array(self.fk(*l0))
        self.target_right = np.array(self.fk(*r0))
        self.target_base = self.localization()

    def _initialize_arrays(self):
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.error = np.zeros(6)
        self.error_pos = np.zeros(3)
        self.error_ori = np.zeros(3)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        
    def _on_ik_mode(self, msg):
        try:
            enabled = bool(msg)
            with self._target_lock:
                self.use_ik = enabled
            print(f"[INFO] IK mode: {'ENABLED' if enabled else 'DISABLED'}")
        except Exception as e:
            print(f"[ERROR] Invalid ik_mode message: {msg}, error: {e}")

    def _on_arm_control(self, msg):
        try:
            raw = np.array(msg, dtype=float)
            if raw.shape != (8,):
                raise ValueError(f"Expected 8 values, got {raw.size}")

            h_min, h_max = -75.0, 75.0   
            h_out_min, h_out_max = 0.0, 1.5  

            a_min, a_max = -30.0, 30.0    
            a_out_min, a_out_max = 0.0, 0.6

            def remap(value, in_min, in_max, out_min, out_max):
                return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)

            l_h1 = np.clip(remap(raw[0], h_min, h_max, h_out_min, h_out_max), h_out_min, h_out_max)
            l_h2 = np.clip(remap(raw[1], h_min, h_max, h_out_min, h_out_max), h_out_min, h_out_max)
            l_a1 = np.clip(remap(raw[2], a_min, a_max, a_out_min, a_out_max), a_out_min, a_out_max)
            l_theta = np.deg2rad(raw[3])  
            r_h1 = np.clip(remap(raw[4], h_min, h_max, h_out_min, h_out_max), h_out_min, h_out_max)
            r_h2 = np.clip(remap(raw[5], h_min, h_max, h_out_min, h_out_max), h_out_min, h_out_max)
            r_a1 = np.clip(remap(raw[6], a_min, a_max, a_out_min, a_out_max), a_out_min, a_out_max)
            r_theta = np.deg2rad(raw[7])  
            mapped = np.array([l_h1, l_h2, l_a1, l_theta, r_h1, r_h2, r_a1, r_theta])

            with self._target_lock:
                self.direct_arm_commands = mapped.copy()
        except Exception as e:
            print(f"[ERROR] Invalid u_control message: {msg}, error: {e}")

    def _on_target_base(self, msg):
        try:
            arr = np.array(msg, dtype=float)
            if arr.shape != (3,):
                raise ValueError(f"Expected shape (3,), got {arr.shape}")
            with self._target_lock:
                self.target_base = arr.copy()
        except Exception as e:
            print(f"[PubSub] Invalid target_base message: {msg}, error: {e}")

    def _on_target_left(self, msg):
        try:
            arr = np.array(msg, dtype=float)
            if arr.shape != (3,):
                raise ValueError(f"Expected shape (3,), got {arr.shape}")
            with self._target_lock:
                self.target_left = arr.copy()
        except Exception as e:
            print(f"[PubSub] Invalid target_left message: {msg}, error: {e}")

    def _on_target_right(self, msg):
        try:
            arr = np.array(msg, dtype=float)
            if arr.shape != (3,):
                raise ValueError(f"Expected shape (3,), got {arr.shape}")
            with self._target_lock:
                self.target_right = arr.copy()
        except Exception as e:
            print(f"[PubSub] Invalid target_right message: {msg}, error: {e}")

    def reset(self, keyframe_name:str):
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        
    def get_keyframe(self, keyframe_name:str):
        def format_array(arr):
            return " ".join(f"{x:.6f}" for x in arr)
        
        print(f'<key\n    name="{keyframe_name}"')
        print(f'    qpos="  {format_array(self.data.qpos)}"')
        print(f'    qvel="  {format_array(self.data.qvel)}"')
        print(f'    ctrl="  {format_array(self.data.ctrl)}"')
        print('/>')
        
    def configure_model(self):
        self.model.opt.timestep = self.DT
        self.model.body_gravcomp[:] = True
        
    def get_joint_qpos_addr(self, joint_name):
        jnt_id = self.model.joint(joint_name).id
        return self.model.jnt_qposadr[jnt_id]
    
    def obstacle_avoidance(self):
        pass
    
    def localization(self):
        x, y = self.data.xpos[self.base_id, 0], self.data.xpos[self.base_id, 1]
        w, xq, yq, zq = self.data.xquat[self.base_id]
        yaw = np.arctan2(2 * (w * zq + xq * yq), 1 - 2 * (yq**2 + zq**2))
        return np.array([x, y, yaw])
    
    def get_encoder(self):
        z1_left  = self.data.qpos[self.get_joint_qpos_addr("ColumnLeftBearingJoint_1")]
        z1_right = self.data.qpos[self.get_joint_qpos_addr("ColumnRightBearingJoint_1")]
        
        z2_left  = self.data.qpos[self.get_joint_qpos_addr("ColumnLeftBearingJoint_2")]
        z2_right = self.data.qpos[self.get_joint_qpos_addr("ColumnRightBearingJoint_2")]
        
        horizontal_1  = self.data.qpos[self.get_joint_qpos_addr("ArmLeftJoint_1")]
        horizontal_2 = self.data.qpos[self.get_joint_qpos_addr("ArmLeftJoint_2")]

        yaw_1 = self.data.qpos[self.get_joint_qpos_addr("BaseJoint_1")]
        yaw_2 = self.data.qpos[self.get_joint_qpos_addr("BaseJoint_2")]
        
        encoder_left = np.array([z1_left, z1_right, horizontal_1, yaw_1])
        encoder_right = np.array([z2_left, z2_right, horizontal_2, yaw_2])
        return encoder_left, encoder_right
        
    def send_command_arm(self, u_control):
        u_control = np.asarray(u_control)
        if u_control.shape != (len(self.actuator_ids),):
            raise ValueError(f"Control input shape {u_control.shape} does not match number of actuators ({len(self.actuator_ids)})")

        ctrl_ranges = self.model.actuator_ctrlrange[self.actuator_ids]  
        lo = ctrl_ranges[:, 0]
        hi = ctrl_ranges[:, 1]
        u_clipped = np.clip(u_control, lo, hi)
        self.data.ctrl[self.actuator_ids] = u_clipped
        
    def fk(self, h1, h2, a1, theta, d2=0.1, l3_max=1.0, eps=1e-9):
        p1 = np.array([0.0, 0.0, h1])
        p2 = np.array([d2 * np.cos(theta), d2 * np.sin(theta), h2])
        vec = p1 - p2
        dist = np.linalg.norm(vec)
        if dist < eps:
            return None, None 
        u = vec / dist
        ee = p1 + a1 * u

        if np.linalg.norm(ee - p1) > l3_max + 1e-9:
            return None, np.degrees(np.arctan2(h2 - h1, d2))
        alpha_deg = np.degrees(np.arctan2(h2 - h1, d2))
        return ee
    
    def ik(self, target, arm="left", d2=0.1, l3_max=0.7, alpha_min_deg=10.0,
        bounds_h=(0.0, 1.5), bounds_a=(0.0, 0.7), tol=1e-6, 
        cache_threshold=0.001):

        target = np.array(target, dtype=float)

        if not hasattr(self, '_ik_cache'):
            self._ik_cache = {
                'left': {'target': None, 'result': None},
                'right': {'target': None, 'result': None}
            }

        cache = self._ik_cache[arm]

        if cache['target'] is not None:
            dist = np.linalg.norm(target - cache['target'])
            if dist <= cache_threshold:
                return cache['result'].copy()

        prev_target = cache['target'] if cache['target'] is not None else np.zeros(3)
        delta_norm = np.linalg.norm(target - prev_target)

        print(f"[IK] Recomputing for {arm} arm: target = {target} (Δ = {delta_norm:.4f} m)")
        
        def cost(vars, w_a1=1e-2):
            h1, h2, a1, theta = vars
            ee = self.fk(h1, h2, a1, theta, d2=d2, l3_max=l3_max)
            if ee is None:
                return 1e3 + 1e2 * np.linalg.norm(np.array([h1, h2, a1]) - 0.5)
            dist_err = np.sum((ee - target)**2)
            return float(dist_err + w_a1 * a1)
        
        def angle_ineq(vars):
            h1, h2, a1, theta = vars
            alpha_deg = np.degrees(np.arctan2(h2 - h1, d2))
            return alpha_deg * alpha_deg - alpha_min_deg * alpha_min_deg

        cons = ({'type': 'ineq', 'fun': angle_ineq},)
        b = [(bounds_h[0], bounds_h[1]),  # h1
            (bounds_h[0], bounds_h[1]),  # h2
            (bounds_a[0], bounds_a[1]),  # a1
            (-np.pi, np.pi)]            # theta

        if arm == "left":
            x0, _ = self.get_encoder()
        elif arm == "right":
            _, x0 = self.get_encoder()
        else:
            x0 = np.array([0.0, 0.0, 0.0, 0.0])
            
        res = minimize(cost, x0, method='SLSQP', bounds=b, constraints=cons,
                    options={'ftol': 1e-9, 'maxiter': 50, 'disp': False})

        if not res.success:
            print(f"[IK] Failed for {arm} arm! Using fallback.")
            if cache['result'] is not None:
                return cache['result'].copy()
            else:
                return x0

        result = np.array([float(res.x[0]), float(res.x[1]), float(res.x[2]), float(res.x[3])])

        self._ik_cache[arm]['target'] = target.copy()
        self._ik_cache[arm]['result'] = result.copy()
        return result
            
    def pid_base_joints(self, target_angle_1, target_angle_2, kp=10, ki=0.0, kd=7):
        dt = self.model.opt.timestep
        jnt1_id = self.model.joint("BaseJoint_1").id
        qpos1 = self.data.qpos[self.model.jnt_qposadr[jnt1_id]]
        error1 = (target_angle_1 - qpos1 + np.pi) % (2 * np.pi) - np.pi

        self.base_integral_1 += error1 * dt
        derivative1 = (error1 - self.base_prev_error_1) / dt 
        self.base_prev_error_1 = error1

        torque1 = kp * error1 + ki * self.base_integral_1 + kd * derivative1

        jnt2_id = self.model.joint("BaseJoint_2").id
        qpos2 = self.data.qpos[self.model.jnt_qposadr[jnt2_id]]
        error2 = (target_angle_2 - qpos2 + np.pi) % (2 * np.pi) - np.pi

        self.base_integral_2 += error2 * dt
        derivative2 = (error2 - self.base_prev_error_2) / dt
        self.base_prev_error_2 = error2

        torque2 = kp * error2 + ki * self.base_integral_2 + kd * derivative2

        return torque1, torque2
                
    def control_base(self, target, alpha=0):
        k_p = 20.0      # was 5.0
        k_i = 0.1
        k_d = 0.8
        
        k_p_theta = 8.0  # was 5.0
        k_i_theta = 0.1
        k_d_theta = 0.8

        self.mobile_dot[0] = self.data.qvel[19]
        self.mobile_dot[1] = self.data.qvel[6] 
        self.mobile_dot[2] = self.data.qvel[45]
        self.mobile_dot[3] = self.data.qvel[32]

        target_x, target_y, target_yaw = target 

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

        MAX_VEL = 15.0
        self.target_vel = np.clip(self.target_vel, -MAX_VEL, MAX_VEL)    

        self.command = self.target_vel - self.mobile_dot
        eps = 0.05
        self.command[np.abs(self.command) < eps] = 0.0

        self.data.ctrl[0] = self.command[1]
        self.data.ctrl[1] = self.command[0]
        self.data.ctrl[2] = self.command[3]
        self.data.ctrl[3] = self.command[2]
        
    def control_arms(self):
        q_left, q_right = self.get_encoder()        
        current_u_left = self.data.ctrl[self.actuator_ids[0:3]]
        current_u_right = self.data.ctrl[self.actuator_ids[4:7]]
        
        if self.use_ik:
            u_left_desired = self.ik(target=self.target_left, arm="left")
            u_right_desired = self.ik(target=self.target_right, arm="right")
        else:
            u_left_desired = self.direct_arm_commands[0:4]   
            u_right_desired = self.direct_arm_commands[4:8] 
            
        u_base_left, u_base_right = self.pid_base_joints(u_left_desired[3], u_right_desired[3])
        
        offset = np.array([-0.0036, -0.0062, -0.0006])
        raw_cmd_L = (u_left_desired[:3] + offset) * 100
        raw_cmd_R = (u_right_desired[:3] + offset) * 100

        alpha = 0.05
        self._smooth_cmd_L = (1 - alpha) * current_u_left + alpha * raw_cmd_L
        self._smooth_cmd_R = (1 - alpha) * current_u_right + alpha * raw_cmd_R

        u_cmd = np.concatenate([
            self._smooth_cmd_L,  
            [u_base_left],      
            self._smooth_cmd_R,  
            [u_base_right]      
        ])
        
        self.send_command_arm(u_cmd)
        
    def step_simulation(self, render=True):
        self.control_base(target=self.target_base, alpha=0.1)
        self.control_arms()

        mujoco.mj_step(self.model, self.data, nstep=10)

        if self.run_mode == "glfw" and render:
            mujoco.mjv_updateScene(
                self.model, self.data, self.opt, None, self.camera, 0xFFFF, self.scene
            )
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
    def camera_display(self):
        self.frame_count = getattr(self, 'frame_count', 0)
        self.renderer_top.update_scene(self.data, self.camera)
        rgb_top = self.renderer_top.render()
        if rgb_top is None or rgb_top.size == 0:
            print("Error: Top view rendering failed")
            return
        bgr_top = cv2.cvtColor(rgb_top, cv2.COLOR_RGB2BGR)

        if self.top_video_writer is not None and self.top_video_writer.isOpened():
            self.top_video_writer.write(bgr_top)

        self.frame_count += 1

        cv2.imshow("MuJoCo Top View", bgr_top)
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

    def run_glfw(self):
        mujoco.mj_step(self.model, self.data, nstep=1)
        while not glfw.window_should_close(self.window) and not self._terminate:
            self.step_simulation()
            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(self.viewport, self.scene, self.ctx)
        glfw.terminate()
           
    def on_key(self, window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.window, True)
            return
        
        if key == glfw.KEY_ENTER:
            self.reset()
            self.gripper_ctrl = 0.0
            return

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
    xml_path = os.path.join(os.path.dirname(__file__), '..', 'env', 'market_world_plain.xml')
    xml_path = os.path.abspath(xml_path)
    sim = ParallelRobot(xml_path, args.run, args.record)
    if args.run == "glfw":
        sim.run_glfw()
    else:  # cv
        sim.run_cv()