import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco, os
import glfw, time
import numpy as np
import datetime, cv2
from scipy.optimize import minimize
import argparse
from modules.trajectory_opt import TrajectoryOptimizer

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
        
    def __init__(self, path: str, run_mode: str, control_mode: str, record: bool):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.reset()
        self._initialize_ids()
        self._initialize_arrays()
        
        self._terminate = False  
        self.paused = False 
        self.run_mode = run_mode.lower()
        self.control_mode = control_mode.lower()
        self.record = record
        self.current_ctrl = np.zeros(len(self.ACTUATOR_NAMES))

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
        
        if self.control_mode == "keyboard":
            self.print_controls()
        
        self.h1, self.h2, self.l1, self.l2 = 0, 0, 0, 0
        self.top_video_writer = None
        self.pov_video_writer = None
        
    def _initialize_ids(self):
        self.arm_link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Arm")
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Gripper_Link1")
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_footprint")

        self.dof_ids = np.array([self.model.joint(name).id for name in self.JOINT_NAMES])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in self.ACTUATOR_NAMES])
        self.gripper_ids_left = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.GRIPPER_ACT_LEFT]
        self.gripper_ids_right = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.GRIPPER_ACT_RIGHT]
        
        l0, r0 = self.get_encoder()
        self.target_left = self.fk(*l0)
        self.target_right = self.fk(*r0)

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
        
    def reset(self):
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        
    def configure_model(self):
        self.model.opt.timestep = self.DT
        self.model.body_gravcomp[:] = True
        
    def get_joint_qpos_addr(self, joint_name):
        jnt_id = self.model.joint(joint_name).id
        return self.model.jnt_qposadr[jnt_id]
    
    def obstacle_avoidance(self):
        pass
    
    def localization(self):
        current_x, current_y = self.data.xpos[self.base_id, 0], self.data.xpos[self.base_id, 1]
        w, x, y, z = self.data.xquat[self.base_id]
        current_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        return np.array([current_x, current_y, current_yaw])
        
    def generate_trajectory(self, n_coeffs, derivatives, waypoints, times):
        optimizer = TrajectoryOptimizer(n_coeffs, derivatives, times)

        states, coeffs = optimizer.generate_trajectory(waypoints, num_points=50)
        total_duration = times[-1] - times[0]
        traj_t = np.linspace(times[0], times[-1], 50)
        
        return states, traj_t
        
    def site_pos_relative_to_body(self, site_name, body_name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        site_pos_world  = self.data.site_xpos[site_id]
        body_pos_world  = self.data.xpos[body_id]
        body_quat_world = self.data.xquat[body_id]  # (w, x, y, z)

        vec_world = site_pos_world - body_pos_world

        w, x, y, z = body_quat_world
        quat_conj = np.array([w, -x, -y, -z])

        def quat_rotate_vec(q, v):
            w, x, y, z = q
            vx, vy, vz = v
            return np.array([
                (1 - 2*y*y - 2*z*z)*vx + (2*x*y - 2*w*z)*vy + (2*x*z + 2*w*y)*vz,
                (2*x*y + 2*w*z)*vx + (1 - 2*x*x - 2*z*z)*vy + (2*y*z - 2*w*x)*vz,
                (2*x*z - 2*w*y)*vx + (2*y*z + 2*w*x)*vy + (1 - 2*x*x - 2*y*y)*vz
            ])

        vec_local = quat_rotate_vec(quat_conj, vec_world)
        return vec_local, vec_world
    
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
            
    def pid_base_joints(self, target_angle_1, target_angle_2, kp=20.0, ki=0.05, kd=2.0):
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
    
    def add_visual_capsule(self, scene, point1, point2, radius, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.zeros(9),
            rgba=np.asarray(rgba, dtype=np.float32)
        )
        mujoco.mjv_connector(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            width=radius,
            from_=np.asarray(point1, dtype=np.float64),
            to=np.asarray(point2, dtype=np.float64)
        )
        scene.ngeom += 1

    def _quat_from_z_to_vec(self, vec):
        vec = np.asarray(vec, dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_quatZ2Vec(quat, vec / norm)
        return quat

    def add_visual_arrow(self, scene, point1, point2, rgba):
        if scene.ngeom >= scene.maxgeom:
            return

        p1 = np.asarray(point1, dtype=np.float64)
        p2 = np.asarray(point2, dtype=np.float64)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-8:
            return

        pos = (p1 + p2) / 2.0

        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_quatZ2Vec(quat, direction / length)
        mat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat)

        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.array([0.05, 0.05, length*2], dtype=np.float32),
            pos=pos.astype(np.float32),
            mat=mat.astype(np.float32),
            rgba=np.asarray(rgba, dtype=np.float32)
        )
        scene.ngeom += 1
                
    def draw_full_trajectory(self, x, y, z=0.05, base_rgba=np.array([0.0, 0.8, 1.0, 1.0])):
        if np.isscalar(z):
            z = np.full_like(x, z)
        n = len(x)
        if n < 2:
            return

        MAX_GEOMS = min(500, self.scene.maxgeom)

        for i in range(n - 1):
            if self.scene.ngeom >= MAX_GEOMS:
                break

            t = i / max(n - 2, 1)  
            alpha = 0.1 + 0.7 * t 
            rgba = base_rgba.copy()
            rgba[3] = alpha 

            p1 = [x[i], y[i], z[i]]
            p2 = [x[i+1], y[i+1], z[i+1]]
            self.add_visual_arrow(self.scene, p1, p2, rgba)

    def draw_blinking_trajectory(self, x, y, z=0.05):
        if np.isscalar(z):
            z = np.full_like(x, z)
        n = len(x)
        if n < 2:
            return

        current_time = time.time()
        if current_time - self.last_blink_time > self.blink_speed:
            self.blink_index = (self.blink_index + 1) % (n - 1)
            self.last_blink_time = current_time

        i = self.blink_index
        p1 = [x[i], y[i], z[i]]
        p2 = [x[i+1], y[i+1], z[i+1]]
        self.add_visual_arrow(self.scene, p1, p2, rgba=[1.0, 0, 0, 1.0])  

        for j in range(n - 1):
            if j == i:
                continue
            p1_bg = [x[j], y[j], z[j]]
            p2_bg = [x[j+1], y[j+1], z[j+1]]
            self.add_visual_arrow(self.scene, p1_bg, p2_bg, rgba=[0, 0, 0, 0.3])
            
    def control_mobile_robot(self, target, alpha=0.0):
        if not hasattr(self, 'integral_x'):
            self.integral_x = self.integral_y = self.integral_yaw = 0.0
            self.prev_delta_x = self.prev_delta_y = self.prev_delta_yaw = 0.0
            self.deriv_x = self.deriv_y = self.deriv_yaw = 0.0

        current_pos = self.localization()  # [x, y, yaw]
        x_curr, y_curr, yaw_curr = current_pos
        x_ref, y_ref, yaw_ref = target

        delta_x = x_ref - x_curr
        delta_y = y_ref - y_curr
        delta_yaw = np.arctan2(np.sin(yaw_ref - yaw_curr), np.cos(yaw_ref - yaw_curr))

        cos_yaw = np.cos(yaw_curr)
        sin_yaw = np.sin(yaw_curr)
        delta_x_local = cos_yaw * delta_x + sin_yaw * delta_y
        delta_y_local = -sin_yaw * delta_x + cos_yaw * delta_y

        dt = self.model.opt.timestep

        self.integral_x += delta_x_local * dt
        self.integral_y += delta_y_local * dt
        self.integral_yaw += delta_yaw * dt

        deriv_x_local = (delta_x_local - getattr(self, 'prev_delta_x', delta_x_local)) / dt
        deriv_y_local = (delta_y_local - getattr(self, 'prev_delta_y', delta_y_local)) / dt
        deriv_yaw = (delta_yaw - getattr(self, 'prev_delta_yaw', delta_yaw)) / dt

        self.deriv_x = alpha * deriv_x_local + (1 - alpha) * getattr(self, 'deriv_x', deriv_x_local)
        self.deriv_y = alpha * deriv_y_local + (1 - alpha) * getattr(self, 'deriv_y', deriv_y_local)
        self.deriv_yaw = alpha * deriv_yaw + (1 - alpha) * getattr(self, 'deriv_yaw', deriv_yaw)

        self.prev_delta_x = delta_x_local
        self.prev_delta_y = delta_y_local
        self.prev_delta_yaw = delta_yaw

        k_p, k_i, k_d = 4.0, 0.05, 0.5
        k_p_theta, k_i_theta, k_d_theta = 4, 0.1, 0.2

        v_x_local = k_p * delta_x_local + k_i * self.integral_x + k_d * self.deriv_x
        v_y_local = k_p * delta_y_local + k_i * self.integral_y + k_d * self.deriv_y
        omega = k_p_theta * delta_yaw + k_i_theta * self.integral_yaw + k_d_theta * self.deriv_yaw

        D, r = self.D, self.r
        vel_FL = (v_x_local - v_y_local - omega * D) / r 
        vel_FR = (v_x_local + v_y_local + omega * D) / r 
        vel_RL = (v_x_local + v_y_local - omega * D) / r 
        vel_RR = (v_x_local - v_y_local + omega * D) / r 

        wheel_vels = np.array([
            self.data.qvel[6],   
            self.data.qvel[19],  
            self.data.qvel[32],  
            self.data.qvel[45],  
        ])

        target_vels = np.array([vel_FL, vel_FR, vel_RL, vel_RR])
        command = target_vels - wheel_vels
        command[np.abs(command) < 0.05] = 0.0
        self.data.ctrl[0] = command[1]  
        self.data.ctrl[1] = command[0]  
        self.data.ctrl[2] = command[3]  
        self.data.ctrl[3] = command[2]  
            
    def control_arms(self, target_left, target_right):
        q_left, q_right = self.get_encoder()        
        current_u_left = self.data.ctrl[self.actuator_ids[0:3]]
        current_u_right = self.data.ctrl[self.actuator_ids[4:7]]
        
        u_left_desired = self.ik(target=target_left, arm="left")
        u_right_desired = self.ik(target=target_right, arm="right")

        u_base_left, u_base_right = self.pid_base_joints(u_left_desired[-1], u_right_desired[-1])
        
        offset = np.array([-0.0036, -0.0062, -0.0006])
        
        raw_cmd_L = (u_left_desired[:-1] + offset) * 100
        raw_cmd_R = (u_right_desired[:-1] + offset) * 100

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
        
    def step_simulation(self):
        current_time = self.data.time

        # ==============================
        # PHASE 1: Initial Base Trajectory (to pick area)
        # ==============================
        if not hasattr(self, 'base_trajectory_planned'):
            current = self.localization()  # [x, y, yaw]
            print("Planning BASE trajectory to pick area:", current)
            x0, y0, yaw0 = current
            waypoints = np.array([
                [x0, y0, yaw0],
                [1.5, -2.5, yaw0 / 2],
                [1.5, -5.3, yaw0 / 3],
                [3.0, -6.2, 0.0],
            ])
            times = [0.0, 5.0, 8.0, 11.0]
            n_coef = [4, 4, 4]
            d_dt = [2, 2, 2]
            states, timestep = self.generate_trajectory(n_coef, d_dt, waypoints, times)
            self.traj_t = timestep
            self.traj_x = states[0][0, :]
            self.traj_y = states[1][0, :]
            self.traj_yaw = states[2][0, :]
            self.base_trajectory_end_time = times[-1]
            self.base_trajectory_planned = True

        # ==============================
        # PHASE 2: Arm Pick & Return
        # ==============================
        target_left = self.target_left[:3]
        roll_arm = 0.0
        pitch_arm = 0.0

        if (not hasattr(self, 'arm_trajectory_planned')) and \
        (current_time >= self.base_trajectory_end_time):

            print("Planning ARM PICK trajectory...")
            current = self.target_left
            if len(current) < 3:
                current = np.append(current, [0.0] * (3 - len(current)))
            self.arm_initial_pose = current[:3].copy()

            arm_l_waypoints = np.array([
                [current[0], current[1], current[2], 0.0],
                [-0.33, -0.053, current[2], 1.12],
                [-0.33, -0.053, -0.01, 1.12],
            ])
            arm_times = [0.0, 2.0, 4.0]
            arm_coef = [5, 5, 5, 5]
            arm_dt = [2, 2, 2, 2]
            arm_states, arm_timestep = self.generate_trajectory(
                arm_coef, arm_dt, arm_l_waypoints, arm_times
            )
            self.arm_traj_t = arm_timestep
            self.arm_traj_x = arm_states[0][0, :]
            self.arm_traj_y = arm_states[1][0, :]
            self.arm_traj_z = arm_states[2][0, :]
            self.arm_traj_roll = arm_states[3][0, :]
            self.arm_trajectory_planned = True
            self.arm_trajectory_end_time = self.base_trajectory_end_time + arm_times[-1]

        if hasattr(self, 'arm_trajectory_planned') and current_time >= self.base_trajectory_end_time:
            if current_time <= self.arm_trajectory_end_time:
                finger_ids = [self.gripper_ids_left[0], self.gripper_ids_left[3], self.gripper_ids_left[6]]
                self.gripper_open_pos = -0.3
                for fid in finger_ids:
                    self.data.ctrl[fid] = self.gripper_open_pos

                t_local = current_time - self.base_trajectory_end_time
                pitch_arm = 1.8
                x_arm = np.interp(t_local, self.arm_traj_t, self.arm_traj_x)
                y_arm = np.interp(t_local, self.arm_traj_t, self.arm_traj_y)
                z_arm = np.interp(t_local, self.arm_traj_t, self.arm_traj_z)
                roll_arm = np.interp(t_local, self.arm_traj_t, self.arm_traj_roll)

                target_left = [x_arm, y_arm, z_arm]

            else:
                if not hasattr(self, 'arm_close_hold_started'):
                    finger_ids = [self.gripper_ids_left[0], self.gripper_ids_left[3], self.gripper_ids_left[6]]
                    self.gripper_closed_pos = 0.3
                    for fid in finger_ids:
                        self.data.ctrl[fid] = self.gripper_closed_pos

                    self.arm_close_hold_start_time = current_time
                    self.arm_close_hold_duration = 2.0
                    self.arm_close_hold_started = True

                hold_elapsed = current_time - self.arm_close_hold_start_time
                if hold_elapsed <= self.arm_close_hold_duration:
                    target_left = [self.arm_traj_x[-1], self.arm_traj_y[-1], self.arm_traj_z[-1]]
                    pitch_arm = 1.8
                    roll_arm = self.arm_traj_roll[-1]

                else:
                    if not hasattr(self, 'arm_return_started'):
                        self.arm_return_start_time = current_time
                        self.arm_return_duration = 4.0
                        self.arm_return_started = True

                    return_elapsed = current_time - self.arm_return_start_time
                    if return_elapsed <= self.arm_return_duration:
                        alpha = return_elapsed / self.arm_return_duration
                        x_ret = (1 - alpha) * self.arm_traj_x[-1] + alpha * self.arm_initial_pose[0]
                        y_ret = (1 - alpha) * self.arm_traj_y[-1] + alpha * self.arm_initial_pose[1]
                        z_ret = (1 - alpha) * self.arm_traj_z[-1] + alpha * self.arm_initial_pose[2]

                        target_left = [x_ret, y_ret, z_ret]
                        pitch_arm = 1.8
                        roll_arm = self.arm_traj_roll[-1]

                    else:
                        target_left = self.arm_initial_pose.tolist()
                        pitch_arm = 1.8
                        roll_arm = self.arm_traj_roll[-1]
                        self.arm_full_return_done = True
                
        # ==============================
        # PHASE 3: Final Base Trajectory (to drop zone)
        # ==============================
        if hasattr(self, 'arm_full_return_done') and not hasattr(self, 'final_base_planned'):
            print("Planning FINAL BASE trajectory to drop zone [3.0, -6.7, -pi/2]...")
            current = self.localization()
            final_waypoints = np.array([
                [current[0], current[1], current[2]],
                [3.0, -6.7, -np.pi / 2]
            ])
            final_times = [0.0, 4.0]
            n_coef = [4, 4, 4]
            d_dt = [2, 2, 2]
            states, timestep = self.generate_trajectory(n_coef, d_dt, final_waypoints, final_times)
            self.final_traj_t = timestep
            self.final_traj_x = states[0][0, :]
            self.final_traj_y = states[1][0, :]
            self.final_traj_yaw = states[2][0, :]
            self.final_base_end_time = current_time + final_times[-1]
            self.final_base_planned = True

        # ==============================
        # PHASE 4: Arm moves while base is driving to drop zone
        # ==============================
        if hasattr(self, 'final_base_planned') and not hasattr(self, 'arm_drive_trajectory_planned'):
            print("Planning ARM DRIVE trajectory while base moves to drop zone")

            start = self.arm_initial_pose
            start_roll = self.arm_traj_roll[-1]

            start_pitch = self.data.ctrl[self.gripper_ids_left[13]] 
            end_roll = start_roll
            end_pitch = 0.00    
            arm_drive_waypoints = np.array([
                [start[0], start[1], start[2], start_roll, start_pitch],
                [-0.6,      0.0,   1.3,      end_roll,  end_pitch]
            ])

            arm_drive_times = [0.0, 2.0]
            arm_drive_coef = [5, 5, 5, 5, 5]
            arm_drive_dt = [2, 2, 2, 2, 2]

            arm_drive_states, arm_drive_timestep = self.generate_trajectory(
                arm_drive_coef, arm_drive_dt, arm_drive_waypoints, arm_drive_times
            )

            self.arm_drive_t = arm_drive_timestep
            self.arm_drive_x = arm_drive_states[0][0, :]
            self.arm_drive_y = arm_drive_states[1][0, :]
            self.arm_drive_z = arm_drive_states[2][0, :]
            self.arm_drive_roll = arm_drive_states[3][0, :]
            self.arm_drive_pitch = arm_drive_states[4][0, :]  

            self.arm_drive_end_time = self.final_base_end_time
            self.arm_drive_trajectory_planned = True


        if hasattr(self, 'arm_drive_trajectory_planned') and current_time <= self.arm_drive_end_time:
            t_drive = current_time - (self.final_base_end_time - 4.0)
            x_arm = np.interp(t_drive, self.arm_drive_t, self.arm_drive_x)
            y_arm = np.interp(t_drive, self.arm_drive_t, self.arm_drive_y)
            z_arm = np.interp(t_drive, self.arm_drive_t, self.arm_drive_z)
            roll_arm = np.interp(t_drive, self.arm_drive_t, self.arm_drive_roll)
            pitch_arm = np.interp(t_drive, self.arm_drive_t, self.arm_drive_pitch) 

            target_left = [x_arm, y_arm, z_arm]


        # ==============================
        # PHASE 5: ARRIVED — drop + open gripper
        # ==============================
        if hasattr(self, 'arm_drive_trajectory_planned') and current_time > self.arm_drive_end_time:
            target_left = [-0.6, 0.0, 1.27]
            roll_arm = self.arm_drive_roll[-1]
            pitch_arm = self.arm_drive_pitch[-1] 
            if not hasattr(self, 'drop_started'):
                print("Opening gripper — Releasing object!")
                self.drop_started = True
                self.drop_start_time = current_time
                self.gripper_open_duration = 2

                finger_ids = [
                    self.gripper_ids_left[0],
                    self.gripper_ids_left[3],
                    self.gripper_ids_left[6],
                ]
                self.gripper_open_pos = -1
                for fid in finger_ids:
                    self.data.ctrl[fid] = self.gripper_open_pos

            elapsed = current_time - self.drop_start_time
            if elapsed >= self.gripper_open_duration:
                self.drop_complete = True

        # ==============================
        # BASE TRAJECTORY REFERENCE
        # ==============================
        if hasattr(self, 'final_base_planned') and current_time >= self.final_base_end_time:
            x_ref, y_ref, yaw_ref = (
                self.final_traj_x[-1], self.final_traj_y[-1], self.final_traj_yaw[-1]
            )
        elif hasattr(self, 'final_base_planned'):
            t_local = current_time - (self.final_base_end_time - 4.0)
            x_ref = np.interp(t_local, self.final_traj_t, self.final_traj_x)
            y_ref = np.interp(t_local, self.final_traj_t, self.final_traj_y)
            yaw_ref = np.interp(t_local, self.final_traj_t, self.final_traj_yaw)
        else:
            if current_time <= self.base_trajectory_end_time:
                x_ref = np.interp(current_time, self.traj_t, self.traj_x)
                y_ref = np.interp(current_time, self.traj_t, self.traj_y)
                yaw_ref = np.interp(current_time, self.traj_t, self.traj_yaw)
            else:
                x_ref, y_ref, yaw_ref = (
                    self.traj_x[-1], self.traj_y[-1], self.traj_yaw[-1]
                )

        # ==============================
        # SEND CONTROL COMMANDS
        # ==============================
        self.control_mobile_robot(target=[x_ref, y_ref, yaw_ref], alpha=0.1)
        self.control_arms(target_left, self.target_right)

        if (hasattr(self, 'arm_trajectory_planned') or hasattr(self, 'drop_arm_planned')) and \
        current_time >= self.base_trajectory_end_time:
            self.data.ctrl[self.gripper_ids_left[14]] = roll_arm
            self.data.ctrl[self.gripper_ids_left[13]] = pitch_arm  

        # ==============================
        # Physics & Rendering
        # ==============================
        mujoco.mj_step(self.model, self.data, nstep=5)

        # if not hasattr(self, 'blink_index'):
        #     self.blink_index = 0
        #     self.blink_speed = 0.08 
        #     self.last_blink_time = time.time()

        if self.run_mode == "cv":
            self.camera_display()
        elif self.run_mode == "glfw":
            mujoco.mjv_updateScene(
                self.model, self.data, self.opt, None, self.camera, 0xFFFF, self.scene
            )
            # if hasattr(self, 'traj_x'):
            #     self.draw_blinking_trajectory(self.traj_x, self.traj_y, z=0.05)
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
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.window, True)
            return
        
        if key == glfw.KEY_ENTER:
            self.reset()
            self.gripper_ctrl = 0.0
            return
        
        if key in (glfw.KEY_Z, glfw.KEY_X, glfw.KEY_C, glfw.KEY_V):
            GRIPPER_OPEN_POS = -1
            GRIPPER_CLOSED_POS = 1.2218
            GRIPPER_INCREMENT = 0.01
            
            if key in (glfw.KEY_Z, glfw.KEY_X):
                # LEFT: Z → Close | X → Open
                act_ids = [
                    self.gripper_ids_left[0],  # finger_c_joint_1_1
                    self.gripper_ids_left[3],  # finger_b_joint_1_1
                    self.gripper_ids_left[6]   # finger_a_joint_1_1
                ]
                increment = GRIPPER_INCREMENT if key == glfw.KEY_Z else -GRIPPER_INCREMENT

            elif key in (glfw.KEY_C, glfw.KEY_V):
                # RIGHT: C → Close | V → Open
                act_ids = [
                    self.gripper_ids_right[0],  # finger_c_joint_1_2
                    self.gripper_ids_right[3],  # finger_b_joint_1_2
                    self.gripper_ids_right[6]   # finger_a_joint_1_2
                ]
                increment = GRIPPER_INCREMENT if key == glfw.KEY_C else -GRIPPER_INCREMENT

            for act_id in act_ids:
                current_val = self.data.ctrl[act_id]
                new_val = np.clip(
                    current_val + increment,
                    GRIPPER_OPEN_POS,
                    GRIPPER_CLOSED_POS
                )
                self.data.ctrl[act_id] = new_val
                        
        if key in (glfw.KEY_W, glfw.KEY_S, glfw.KEY_UP, glfw.KEY_DOWN):
            BEARING_INCREMENT = 0.01  
            BEARING_MIN = -1.57       
            BEARING_MAX = 1.57        

            if key == glfw.KEY_W:
                # LEFT: W → increase
                current = self.data.ctrl[self.gripper_ids_left[14]]
                new_val = np.clip(current + BEARING_INCREMENT, BEARING_MIN, BEARING_MAX)
                self.data.ctrl[self.gripper_ids_left[14]] = new_val
                print(f"EE BEARING : {new_val}")

            elif key == glfw.KEY_S:
                # LEFT: S → decrease
                current = self.data.ctrl[self.gripper_ids_left[14]]
                new_val = np.clip(current - BEARING_INCREMENT, BEARING_MIN, BEARING_MAX)
                self.data.ctrl[self.gripper_ids_left[14]] = new_val
                print(f"EE BEARING : {new_val}")

            elif key == glfw.KEY_UP:
                # RIGHT: UP → increase
                current = self.data.ctrl[self.gripper_ids_right[14]]
                new_val = np.clip(current + BEARING_INCREMENT, BEARING_MIN, BEARING_MAX)
                self.data.ctrl[self.gripper_ids_right[14]] = new_val

            elif key == glfw.KEY_DOWN:
                # RIGHT: DOWN → decrease
                current = self.data.ctrl[self.gripper_ids_right[14]]
                new_val = np.clip(current - BEARING_INCREMENT, BEARING_MIN, BEARING_MAX)
                self.data.ctrl[self.gripper_ids_right[14]] = new_val
                
        if key in (glfw.KEY_A, glfw.KEY_D, glfw.KEY_LEFT, glfw.KEY_RIGHT):
            WRIST_Z_INCREMENT = 0.05
            WRIST_Z_MIN = -3.14159
            WRIST_Z_MAX = 3.14159

            if key == glfw.KEY_A:
                # LEFT: A → increase (counter-clockwise)
                current = self.data.ctrl[self.gripper_ids_left[13]]
                new_val = np.clip(current + WRIST_Z_INCREMENT, WRIST_Z_MIN, WRIST_Z_MAX)
                self.data.ctrl[self.gripper_ids_left[13]] = new_val

            elif key == glfw.KEY_D:
                # LEFT: D → decrease (clockwise)
                current = self.data.ctrl[self.gripper_ids_left[13]]
                new_val = np.clip(current - WRIST_Z_INCREMENT, WRIST_Z_MIN, WRIST_Z_MAX)
                self.data.ctrl[self.gripper_ids_left[13]] = new_val

            elif key == glfw.KEY_LEFT:
                # RIGHT: LEFT ARROW → increase (counter-clockwise)
                current = self.data.ctrl[self.gripper_ids_right[13]]
                new_val = np.clip(current + WRIST_Z_INCREMENT, WRIST_Z_MIN, WRIST_Z_MAX)
                self.data.ctrl[self.gripper_ids_right[13]] = new_val

            elif key == glfw.KEY_RIGHT:
                # RIGHT: RIGHT ARROW → decrease (clockwise)
                current = self.data.ctrl[self.gripper_ids_right[13]]
                new_val = np.clip(current - WRIST_Z_INCREMENT, WRIST_Z_MIN, WRIST_Z_MAX)
                self.data.ctrl[self.gripper_ids_right[13]] = new_val

        if key in (
            glfw.KEY_R, glfw.KEY_T, glfw.KEY_Y, glfw.KEY_F, glfw.KEY_G, glfw.KEY_H,
            glfw.KEY_U, glfw.KEY_I, glfw.KEY_O, glfw.KEY_J, glfw.KEY_K, glfw.KEY_L
        ):
            EE_INCREMENT = 0.01  # 2 cm per press — tune as needed
            # === LEFT ARM ===
            if key == glfw.KEY_R:      # +x
                self.target_left[0] += EE_INCREMENT
            elif key == glfw.KEY_T:    # -x
                self.target_left[0] -= EE_INCREMENT
            elif key == glfw.KEY_F:    # +y
                self.target_left[1] += EE_INCREMENT
            elif key == glfw.KEY_G:    # -y
                self.target_left[1] -= EE_INCREMENT
            elif key == glfw.KEY_Y:    # +z
                self.target_left[2] += EE_INCREMENT
            elif key == glfw.KEY_H:    # -z
                self.target_left[2] -= EE_INCREMENT
            # === RIGHT ARM ===
            elif key == glfw.KEY_U:    # +x
                self.target_right[0] += EE_INCREMENT
            elif key == glfw.KEY_I:    # -x
                self.target_right[0] -= EE_INCREMENT
            elif key == glfw.KEY_J:    # +y
                self.target_right[1] += EE_INCREMENT
            elif key == glfw.KEY_K:    # -y
                self.target_right[1] -= EE_INCREMENT
            elif key == glfw.KEY_O:    # +z
                self.target_right[2] += EE_INCREMENT
            elif key == glfw.KEY_L:    # -z
                self.target_right[2] -= EE_INCREMENT
            print(f"Left target: {self.target_left}")
            print(f"Right target: {self.target_right}")


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
    parser.add_argument("--control", choices=["trajectory", "keyboard"], default="trajectory",
                        help="Control mode: 'trajectory' or 'keyboard'")
    parser.add_argument("--record", action="store_true",
                        help="Record video output to MP4 (only applicable with --run cv)")
    args = parser.parse_args()

    if args.record and args.run != "cv":
        print("Warning: --record is only applicable with --run cv. Ignoring --record.")
        args.record = False
    xml_path = os.path.join(os.path.dirname(__file__), '..', 'env', 'market_world.xml')
    xml_path = os.path.abspath(xml_path)
    sim = ParallelRobot(xml_path, args.run, args.control, args.record)
    if args.run == "glfw":
        sim.run_glfw()
    else:  # cv
        sim.run_cv()