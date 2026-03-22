"""
play_m1.py  –  Milestone 1 + 2 GUI
Mobile Manipulator Pick & Place — MuJoCo / GLFW / ImGui

M1: OMPL RRT* navigation
M2: Direct-joint grasping with RL policy + proximity-attach
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import threading
import time
import math

import mujoco
import glfw
import numpy as np
import imgui
from imgui.integrations.glfw import GlfwRenderer

from simulations.morph_i_free_move import ParallelRobot
from modules.pubsub import IPCPubSub
from navigation.ompl_windows_bridge import InProcessNavigator
from navigation.grasp_controller import GraspController

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_OBJECTS     = 10
NUM_SHELF_SLOTS = 10

OBJECT_LABELS = [
    "Obj-0  [Brisk Can]",
    "Obj-1  [Pepsi Max]",
    "Obj-2  [Nescafe Dolce]",
    "Obj-3  [Nescafe Taster]",
    "Obj-4  [Nestle Candy]",
    "Obj-5  [Pepsi Cherry]",
    "Obj-6  [Crunch Bar]",
    "Obj-7  [Skinny Cow]",
    "Obj-8  [Vans Cereal]",
    "Obj-9  [Pepsi Diet]",
]

OBJECT_COLORS_RGBA = [
    (0.9, 0.1, 0.1, 1.0),
    (0.9, 0.5, 0.0, 1.0),
    (0.9, 0.9, 0.0, 1.0),
    (0.1, 0.8, 0.1, 1.0),
    (0.0, 0.8, 0.9, 1.0),
    (0.1, 0.2, 0.9, 1.0),
    (0.6, 0.0, 0.8, 1.0),
    (0.9, 0.3, 0.6, 1.0),
    (0.95, 0.95, 0.95, 1.0),
    (0.5, 0.25, 0.0, 1.0),
]

SHELF_SLOT_POSITIONS = np.array([
    [2.5, -3.5, 0.50],
    [3.0, -3.5, 0.50],
    [3.5, -3.5, 0.50],
    [4.0, -3.5, 0.50],
    [4.5, -3.5, 0.50],
    [2.5, -3.5, 1.00],
    [3.0, -3.5, 1.00],
    [3.5, -3.5, 1.00],
    [4.0, -3.5, 1.50],
    [4.5, -3.5, 1.50],
], dtype=float)

SHELF_SLOT_LABELS = [
    f"Slot-{i}  [row={'low' if SHELF_SLOT_POSITIONS[i,2]<0.75 else ('mid' if SHELF_SLOT_POSITIONS[i,2]<1.25 else 'high')}"
    f"  z={SHELF_SLOT_POSITIONS[i,2]:.2f}m]"
    for i in range(NUM_SHELF_SLOTS)
]

FLOOR_X_RANGE      = (0.5, 7.5)
FLOOR_Y_RANGE      = (-7.8, -6.2)
FLOOR_X_RANGE_2    = (0.5, 7.5)
FLOOR_Y_RANGE_2    = (-3.2, -2.2)
OBJECT_Z           = 0.075
MIN_OBJ_SEPARATION = 0.35

OBJ_RADIUS_RANGE = (0.04, 0.10)
OBJ_HEIGHT_RANGE = (0.06, 0.18)

WAYPOINT_REACH_DIST = 0.50
GOAL_REACH_DIST     = 0.55
WAYPOINT_TIMEOUT    = 180.0


# ---------------------------------------------------------------------------
# Object helpers
# ---------------------------------------------------------------------------

def random_floor_positions(n, rng):
    positions = []
    for idx in range(n):
        use_zone2 = rng.random() < 0.4
        xr = FLOOR_X_RANGE_2 if use_zone2 else FLOOR_X_RANGE
        yr = FLOOR_Y_RANGE_2  if use_zone2 else FLOOR_Y_RANGE
        for _ in range(2000):
            x = rng.uniform(*xr)
            y = rng.uniform(*yr)
            candidate = np.array([x, y])
            if all(np.linalg.norm(candidate - p) >= MIN_OBJ_SEPARATION for p in positions):
                positions.append(candidate)
                break
        else:
            positions.append(np.array([rng.uniform(*FLOOR_X_RANGE),
                                        rng.uniform(*FLOOR_Y_RANGE)]))
    return np.array(positions)


def get_object_qpos_slice(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jnt_id  = model.body_jntadr[body_id]
    if jnt_id < 0:
        return None
    return model.jnt_qposadr[jnt_id]


def randomize_object_positions(model, data, rng):
    xy = random_floor_positions(NUM_OBJECTS, rng)
    for i in range(NUM_OBJECTS):
        qs = get_object_qpos_slice(model, f"pickup_obj_{i}")
        if qs is None:
            continue
        radius = rng.uniform(*OBJ_RADIUS_RANGE)
        height = rng.uniform(*OBJ_HEIGHT_RANGE)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"pickup_obj_{i}")
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == body_id:
                model.geom_size[g] = [radius, height, 0]
                break
        data.qpos[qs:qs+3]   = [xy[i,0], xy[i,1], height]
        data.qpos[qs+3]      = 1.0
        data.qpos[qs+4:qs+7] = 0.0
    mujoco.mj_forward(model, data)
    print(f"[INIT] Randomized {NUM_OBJECTS} objects with random sizes.")


def get_object_world_pos(model, data, obj_idx):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"pickup_obj_{obj_idx}")
    return data.xpos[body_id].copy()


def get_object_geom_ids(model):
    ids = []
    for i in range(NUM_OBJECTS):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"pickup_obj_{i}")
        body_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
        ids.append(body_geoms)
    return ids


def world_to_screen(world_pos, scene, viewport, fovy_deg=45.0):
    cam     = scene.camera[0]
    cam_pos = np.array(cam.pos)
    forward = np.array(cam.forward)
    up      = np.array(cam.up)
    right   = np.cross(forward, up)
    norm    = np.linalg.norm(right)
    if norm < 1e-9:
        return None
    right /= norm
    to_pt = np.array(world_pos, dtype=float) - cam_pos
    cx    =  np.dot(to_pt, right)
    cy    =  np.dot(to_pt, up)
    cz    =  np.dot(to_pt, forward)
    if cz <= 0.01:
        return None
    half_h = math.tan(math.radians(fovy_deg / 2.0))
    aspect  = viewport.width / max(viewport.height, 1)
    ndc_x =  (cx / cz) / (half_h * aspect)
    ndc_y =  (cy / cz) / half_h
    sx = int((ndc_x * 0.5 + 0.5) * viewport.width)
    sy = int((1.0 - (ndc_y * 0.5 + 0.5)) * viewport.height)
    if not (0 <= sx < viewport.width and 0 <= sy < viewport.height):
        return None
    return sx, sy


# ---------------------------------------------------------------------------
# Joystick widget
# ---------------------------------------------------------------------------

class Joystick:
    def __init__(self, inner_radius=50, padding=20, ring_width=20, dead_zone=0.1):
        self.inner_radius      = inner_radius
        self.padding           = padding
        self.ring_width        = ring_width
        self.outer_radius      = inner_radius + padding + ring_width
        self.dead_zone         = dead_zone
        self.xy_value          = np.array([0.0, 0.0])
        self.yaw_drag_value    = 0.0
        self.current_robot_yaw = 0.0
        self.is_active         = False
        self._dragging         = False
        self._mode             = None

    def update_robot_yaw(self, yaw_rad):
        self.current_robot_yaw = float(yaw_rad)

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @property
    def value(self):
        if self._dragging and self._mode == 'yaw':
            return self.xy_value, self._normalize_angle(-self.yaw_drag_value)
        return self.xy_value, None

    def draw(self, label="Joystick"):
        pos      = imgui.get_cursor_screen_pos()
        center_x = pos.x + self.outer_radius
        center_y = pos.y + self.outer_radius
        dl       = imgui.get_window_draw_list()
        dl.add_circle_filled(center_x, center_y, self.outer_radius,
                             imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0))
        dl.add_circle_filled(center_x, center_y, self.inner_radius + self.padding,
                             imgui.get_color_u32_rgba(0.10, 0.10, 0.10, 1.0))
        dl.add_circle_filled(center_x, center_y, self.inner_radius,
                             imgui.get_color_u32_rgba(0.25, 0.25, 0.25, 1.0))
        io         = imgui.get_io()
        mouse_x, mouse_y = io.mouse_pos
        mouse_down = io.mouse_down[0]
        if not self._dragging:
            dx0   = mouse_x - center_x
            dy0   = mouse_y - center_y
            dist0 = (dx0**2 + dy0**2)**0.5
            if dist0 <= self.outer_radius and mouse_down:
                if dist0 <= self.inner_radius:
                    self._mode = 'xy';  self._dragging = True
                elif dist0 >= self.inner_radius + self.padding:
                    self._mode = 'yaw'; self._dragging = True
        dx = dy = 0.0
        if self._dragging and self._mode:
            dx = mouse_x - center_x
            dy = mouse_y - center_y
            if self._mode == 'xy':
                dist = (dx**2 + dy**2)**0.5
                if dist > self.inner_radius:
                    dx = dx * self.inner_radius / dist
                    dy = dy * self.inner_radius / dist
                nx  = dx / self.inner_radius
                ny  = -dy / self.inner_radius
                mag = (nx**2 + ny**2)**0.5
                if mag <= self.dead_zone:
                    nx = ny = 0.0
                else:
                    scale = min(1.0, (mag - self.dead_zone) / (1.0 - self.dead_zone))
                    if mag > 0:
                        nx = nx * scale / mag
                        ny = ny * scale / mag
                self.xy_value = np.array([nx, ny])
            elif self._mode == 'yaw':
                self.yaw_drag_value = np.arctan2(dy, dx)
            self.is_active = True
            if not mouse_down:
                self._dragging = False
                self._mode     = None
                self.xy_value  = np.array([0.0, 0.0])
                self.is_active = False
        else:
            self.xy_value  = np.array([0.0, 0.0])
            self.is_active = False
        knob_x = center_x + (dx if self._mode == 'xy' and self._dragging else 0)
        knob_y = center_y + (dy if self._mode == 'xy' and self._dragging else 0)
        dl.add_circle_filled(knob_x, knob_y, 8,
            imgui.get_color_u32_rgba(0.0, 0.8, 1.0,
                1.0 if (self._mode == 'xy' and self._dragging) else 0.6))
        mirrored_yaw = self._normalize_angle(-self.current_robot_yaw)
        rr = self.inner_radius + self.padding + self.ring_width / 2
        dl.add_circle_filled(
            center_x + rr * np.cos(mirrored_yaw),
            center_y + rr * np.sin(mirrored_yaw),
            6, imgui.get_color_u32_rgba(1.0, 0.7, 0.0, 1.0))
        imgui.invisible_button(label, self.outer_radius * 2, self.outer_radius * 2)
        return self.xy_value, (
            self._normalize_angle(-self.yaw_drag_value)
            if (self._dragging and self._mode == 'yaw') else None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    xml_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'env', 'market_world_m1.xml'))
    sim = ParallelRobot(xml_path, run_mode="glfw", record=False)

    # ── Navigator + Grasp controller ──────────────────────────────────────
    nav        = InProcessNavigator(sim)
    grasp_ctrl = GraspController(sim)

    move_in_progress_flag  = [False]
    grasp_in_progress_flag = [False]
    _current_obj_idx       = [0]

    def _on_nav_complete(success):
        move_in_progress_flag[0] = False
        print(f"[GUI] Navigation complete — success={success}")
        if success:
            obj_world = get_object_world_pos(sim.model, sim.data, _current_obj_idx[0])
            grasp_in_progress_flag[0] = True
            grasp_ctrl.grasp(
                _current_obj_idx[0], obj_world,
                on_complete=_on_grasp_complete)
        else:
            print("[GUI] Navigation failed — skipping grasp")

    def _on_grasp_complete(success):
        grasp_in_progress_flag[0] = False
        print(f"[GUI] Grasp complete — success={success}")

    glfw.make_context_current(sim.window)
    glfw.swap_interval(1)
    glfw.poll_events()
    if not glfw.get_current_context():
        glfw.make_context_current(sim.window)
    imgui.create_context()
    impl = GlfwRenderer(sim.window, attach_callbacks=False)

    obj_geom_ids  = get_object_geom_ids(sim.model)
    obj_rgba_orig = {}
    for i, geom_list in enumerate(obj_geom_ids):
        obj_rgba_orig[i] = [sim.model.geom_rgba[gid].copy() for gid in geom_list]

    sim.opt.label = mujoco.mjtLabel.mjLABEL_SITE
    rng = np.random.default_rng()
    randomize_object_positions(sim.model, sim.data, rng)

    initial_x, initial_y, initial_yaw = sim.localization()
    target_x   = initial_x
    target_y   = initial_y
    target_yaw = initial_yaw

    joystick = Joystick(inner_radius=50, padding=20, ring_width=20, dead_zone=0.1)
    joystick.update_robot_yaw(initial_yaw)

    paused = False
    dt     = sim.model.opt.timestep

    sim.camera.distance  = 6.0
    sim.camera.azimuth   = 90
    sim.camera.elevation = -45
    sim.camera.lookat[:] = [4.0, -5.0, 0.5]

    xpos, ypos = glfw.get_cursor_pos(sim.window)
    sim._last_mouse_x = xpos
    sim._last_mouse_y = ypos

    selected_object = 0
    selected_shelf  = 0
    move_status     = ""

    while not glfw.window_should_close(sim.window):
        glfw.poll_events()
        impl.process_inputs()

        io = imgui.get_io()
        if not io.want_capture_mouse:
            xpos, ypos = glfw.get_cursor_pos(sim.window)
            dx = xpos - sim._last_mouse_x
            dy = ypos - sim._last_mouse_y
            sim._last_mouse_x = xpos
            sim._last_mouse_y = ypos
            factor = 0.001
            left   = glfw.get_mouse_button(sim.window, glfw.MOUSE_BUTTON_LEFT)   == glfw.PRESS
            right  = glfw.get_mouse_button(sim.window, glfw.MOUSE_BUTTON_RIGHT)  == glfw.PRESS
            middle = glfw.get_mouse_button(sim.window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
            if left:
                mujoco.mjv_moveCamera(sim.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                                      dx*factor, dy*factor, sim.scene, sim.camera)
            elif right:
                mujoco.mjv_moveCamera(sim.model, mujoco.mjtMouse.mjMOUSE_MOVE_H,
                                      dx*factor, dy*factor, sim.scene, sim.camera)
            elif middle:
                mujoco.mjv_moveCamera(sim.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                                      dx*factor, dy*factor, sim.scene, sim.camera)
        else:
            xpos, ypos = glfw.get_cursor_pos(sim.window)
            sim._last_mouse_x = xpos
            sim._last_mouse_y = ypos

        actual_x, actual_y, actual_yaw = sim.localization()
        joystick.update_robot_yaw(actual_yaw)

        if not paused:
            (x_local, y_local), yaw_command = joystick.value
            if yaw_command is not None:
                target_yaw = yaw_command
            cos_a    = np.cos(actual_yaw)
            sin_a    = np.sin(actual_yaw)
            world_vx =  cos_a * y_local + sin_a * x_local
            world_vy =  sin_a * y_local - cos_a * x_local
            target_x += world_vx * 1.0 * dt
            target_y += world_vy * 1.0 * dt

            if not move_in_progress_flag[0] and not grasp_in_progress_flag[0]:
                with sim._target_lock:
                    sim.target_base = np.array([target_x, target_y, target_yaw])

            # KEY: teleport held object every step
            grasp_ctrl.update_held_object()

            sim.step_simulation(render=False)

        # Object highlight
        for i, geom_list in enumerate(obj_geom_ids):
            for j, gid in enumerate(geom_list):
                orig = obj_rgba_orig[i][j]
                if i == selected_object:
                    sim.model.geom_rgba[gid] = np.clip(
                        orig[:3] * 1.4 + np.array([0.25, 0.25, 0.25]), 0, 1
                    ).tolist() + [1.0]
                else:
                    sim.model.geom_rgba[gid] = [0.0, 0.0, 0.0, 0.0]

        fb_width, fb_height = glfw.get_framebuffer_size(sim.window)
        sim.viewport.width  = fb_width
        sim.viewport.height = fb_height
        mujoco.mjv_updateScene(sim.model, sim.data, sim.opt, None,
                               sim.camera, mujoco.mjtCatBit.mjCAT_ALL, sim.scene)
        mujoco.mjr_render(sim.viewport, sim.scene, sim.ctx)

        imgui.new_frame()
        bg = imgui.get_background_draw_list()

        for i in range(NUM_OBJECTS):
            pos3d = get_object_world_pos(sim.model, sim.data, i)
            sc    = world_to_screen(pos3d + np.array([0, 0, 0.18]), sim.scene, sim.viewport)
            if sc is None: continue
            sx, sy = sc
            col    = OBJECT_COLORS_RGBA[i]
            is_sel = (i == selected_object)
            radius = 13 if is_sel else 9
            bg.add_circle_filled(sx, sy, radius,
                imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.95) if is_sel
                else imgui.get_color_u32_rgba(col[0]*0.7, col[1]*0.7, col[2]*0.7, 0.90))
            txt_col = imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 1.0) if is_sel \
                      else imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
            bg.add_text(sx - 4, sy - 7, txt_col, str(i))
            if is_sel:
                bg.add_circle(sx, sy, radius + 3,
                    imgui.get_color_u32_rgba(1.0, 0.9, 0.0, 1.0), thickness=2)

        for i in range(NUM_SHELF_SLOTS):
            sp  = SHELF_SLOT_POSITIONS[i]
            sc  = world_to_screen(sp + np.array([0, 0, 0.12]), sim.scene, sim.viewport)
            if sc is None: continue
            sx, sy  = sc
            is_sel  = (i == selected_shelf)
            radius  = 11 if is_sel else 7
            bg.add_circle_filled(sx, sy, radius,
                imgui.get_color_u32_rgba(0.2, 0.7, 0.2, 0.90) if is_sel
                else imgui.get_color_u32_rgba(0.15, 0.45, 0.15, 0.75))
            bg.add_text(sx - 4, sy - 7, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0), str(i))
            if is_sel:
                bg.add_circle(sx, sy, radius + 3,
                    imgui.get_color_u32_rgba(0.0, 1.0, 0.5, 1.0), thickness=2)

        # Bottom status bar
        bar_h = 32
        bar_y = fb_height - bar_h
        bar_w = fb_width
        bg.add_rect_filled(0, bar_y, bar_w, fb_height,
                           imgui.get_color_u32_rgba(0.05, 0.05, 0.15, 0.82))
        col_o = OBJECT_COLORS_RGBA[selected_object]
        if move_in_progress_flag[0]:
            status_txt = "● MOVING"
            status_col = imgui.get_color_u32_rgba(1.0, 0.8, 0.0, 1.0)
        elif grasp_in_progress_flag[0]:
            status_txt = "● GRASPING"
            status_col = imgui.get_color_u32_rgba(1.0, 0.5, 0.0, 1.0)
        elif grasp_ctrl.is_holding():
            status_txt = "● HOLDING"
            status_col = imgui.get_color_u32_rgba(0.3, 1.0, 0.5, 1.0)
        else:
            status_txt = "IDLE"
            status_col = imgui.get_color_u32_rgba(0.5, 0.9, 0.5, 1.0)
        bg.add_text(10,  bar_y + 8,
                    imgui.get_color_u32_rgba(col_o[0], col_o[1], col_o[2], 1.0),
                    f"Object: {OBJECT_LABELS[selected_object]}")
        bg.add_text(280, bar_y + 8,
                    imgui.get_color_u32_rgba(0.3, 0.9, 0.4, 1.0),
                    f"Shelf: {SHELF_SLOT_LABELS[selected_shelf]}")
        bg.add_text(bar_w - 110, bar_y + 8, status_col, status_txt)

        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_bg_alpha(0.95)
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                 imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE |
                 imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        imgui.begin("Robot Control", flags=flags)

        ik_enabled = sim.use_ik
        clicked, ik_enabled = imgui.checkbox("IK Control", ik_enabled)
        if clicked:
            with sim._target_lock:
                sim.use_ik = ik_enabled
                if ik_enabled:
                    l_enc, r_enc = sim.get_encoder()
                    fk_l = sim.fk(*l_enc)
                    fk_r = sim.fk(*r_enc)
                    sim.target_left  = np.array(fk_l) if fk_l else np.array([ 0.5, 0.0, 0.5])
                    sim.target_right = np.array(fk_r) if fk_r else np.array([-0.5, 0.0, 0.5])
                else:
                    l_enc, r_enc = sim.get_encoder()
                    sim.direct_arm_commands[0:4] = l_enc
                    sim.direct_arm_commands[4:8] = r_enc

        imgui.separator(); imgui.text("ARM1")
        imgui.push_item_width(180)
        if ik_enabled:
            tl = list(sim.target_left)
            _, x = imgui.slider_float("X##ARM1", tl[0], -1.0, 1.0, "%.3f")
            _, y = imgui.slider_float("Y##ARM1", tl[1], -1.0, 1.0, "%.3f")
            _, z = imgui.slider_float("Z##ARM1", tl[2], -1.0, 1.0, "%.3f")
            if x != tl[0] or y != tl[1] or z != tl[2]:
                with sim._target_lock:
                    sim.target_left = np.array([x, y, z])
        else:
            h1i,h2i,a1i,th = sim.direct_arm_commands[0:4]
            _,h1u = imgui.slider_float("H1##ARM1", np.interp(h1i,[0,1.5],[-75,75]), -75.0, 75.0, "%.1f")
            _,h2u = imgui.slider_float("H2##ARM1", np.interp(h2i,[0,1.5],[-75,75]), -75.0, 75.0, "%.1f")
            _,a1u = imgui.slider_float("A1##ARM1", np.interp(a1i,[0,0.7],[-45,45]), -45.0, 45.0, "%.1f")
            _,thd = imgui.slider_float("TH##ARM1", np.degrees(th), -90.0, 90.0, "%.1f")
            with sim._target_lock:
                sim.direct_arm_commands[0:4] = [np.interp(h1u,[-75,75],[0,1.5]),
                    np.interp(h2u,[-75,75],[0,1.5]), np.interp(a1u,[-45,45],[0,0.7]),
                    np.radians(thd)]
        imgui.pop_item_width()

        imgui.separator(); imgui.text("ARM2")
        imgui.push_item_width(180)
        if ik_enabled:
            tr = list(sim.target_right)
            _, x = imgui.slider_float("X##ARM2", tr[0], -1.0, 1.0, "%.3f")
            _, y = imgui.slider_float("Y##ARM2", tr[1], -1.0, 1.0, "%.3f")
            _, z = imgui.slider_float("Z##ARM2", tr[2], -1.0, 1.0, "%.3f")
            if x != tr[0] or y != tr[1] or z != tr[2]:
                with sim._target_lock:
                    sim.target_right = np.array([x, y, z])
        else:
            h1i,h2i,a1i,th = sim.direct_arm_commands[4:8]
            _,h1u = imgui.slider_float("H1##ARM2", np.interp(h1i,[0,1.5],[-75,75]), -75.0, 75.0, "%.1f")
            _,h2u = imgui.slider_float("H2##ARM2", np.interp(h2i,[0,1.5],[-75,75]), -75.0, 75.0, "%.1f")
            _,a1u = imgui.slider_float("A1##ARM2", np.interp(a1i,[0,0.7],[-45,45]), -45.0, 45.0, "%.1f")
            _,thd = imgui.slider_float("TH##ARM2", np.degrees(th), -90.0, 90.0, "%.1f")
            with sim._target_lock:
                sim.direct_arm_commands[4:8] = [np.interp(h1u,[-75,75],[0,1.5]),
                    np.interp(h2u,[-75,75],[0,1.5]), np.interp(a1u,[-45,45],[0,0.7]),
                    np.radians(thd)]
        imgui.pop_item_width()

        imgui.separator(); imgui.text("Grippers")
        imgui.text_disabled("(0% = Open, 100% = Closed)")
        OPEN, CLOSED = -1.0, 0.8
        imgui.push_item_width(180)
        if len(sim.gripper_ids_left) >= 7:
            cur = sim.data.ctrl[sim.gripper_ids_left[0]]
            pct = np.clip((cur - OPEN) / (CLOSED - OPEN) * 100.0, 0, 100)
            chg, pct = imgui.slider_float("Left##GRIP", pct, 0.0, 100.0, "%.0f%%")
            if chg:
                v = OPEN + (CLOSED - OPEN) * pct / 100.0
                for idx in [0,3,6]: sim.data.ctrl[sim.gripper_ids_left[idx]] = v
        if len(sim.gripper_ids_right) >= 7:
            cur = sim.data.ctrl[sim.gripper_ids_right[0]]
            pct = np.clip((cur - OPEN) / (CLOSED - OPEN) * 100.0, 0, 100)
            chg, pct = imgui.slider_float("Right##GRIP", pct, 0.0, 100.0, "%.0f%%")
            if chg:
                v = OPEN + (CLOSED - OPEN) * pct / 100.0
                for idx in [0,3,6]: sim.data.ctrl[sim.gripper_ids_right[idx]] = v
        imgui.pop_item_width()

        imgui.separator()
        _, paused = imgui.checkbox("Pause", paused)
        imgui.same_line()
        if imgui.button("Reset Robot"):
            sim.reset("home")
            grasp_ctrl.cancel()
            rx, ry, ryaw = sim.localization()
            target_x, target_y, target_yaw = rx, ry, ryaw
            joystick.update_robot_yaw(ryaw)
            move_in_progress_flag[0]  = False
            grasp_in_progress_flag[0] = False
            with sim._target_lock:
                sim.target_base = np.array([rx, ry, ryaw])
        imgui.same_line()
        if imgui.button("Respawn Objects"):
            grasp_ctrl.cancel()
            randomize_object_positions(sim.model, sim.data, rng)
            move_status = "Objects respawned."

        imgui.separator()
        imgui.text("Joystick  (inner=translate, ring=rotate):")
        joystick.draw("MainJoystick")
        imgui.text(f"Base → X:{target_x:.2f}  Y:{target_y:.2f}  Yaw:{np.degrees(target_yaw):.1f}°")

        # PICK & PLACE PANEL
        imgui.separator(); imgui.separator()
        dl    = imgui.get_window_draw_list()
        p     = imgui.get_cursor_screen_pos()
        win_w = imgui.get_window_width() - imgui.get_style().window_padding.x * 2
        dl.add_rect_filled(p.x, p.y, p.x + win_w, p.y + 20,
                           imgui.get_color_u32_rgba(0.15, 0.35, 0.55, 1.0), rounding=3.0)
        imgui.set_cursor_screen_pos((p.x + 6, p.y + 3))
        imgui.text_colored("PICK & PLACE", 1.0, 1.0, 1.0, 1.0)
        imgui.dummy(0, 4)

        imgui.push_item_width(230)
        imgui.text("Object ID")
        imgui.same_line()
        col = OBJECT_COLORS_RGBA[selected_object]
        imgui.color_button(f"##sw{selected_object}", col[0], col[1], col[2], col[3],
                           flags=0, width=14, height=14)
        _, selected_object = imgui.combo("##ObjCombo", selected_object, OBJECT_LABELS)
        obj_pos = get_object_world_pos(sim.model, sim.data, selected_object)
        imgui.text_disabled(f"  Pos: ({obj_pos[0]:.2f}, {obj_pos[1]:.2f}, {obj_pos[2]:.2f})")
        imgui.spacing()
        imgui.text("Shelf Slot ID")
        _, selected_shelf = imgui.combo("##ShelfCombo", selected_shelf, SHELF_SLOT_LABELS)
        sp = SHELF_SLOT_POSITIONS[selected_shelf]
        imgui.text_disabled(f"  Pos: ({sp[0]:.2f}, {sp[1]:.2f}, {sp[2]:.2f})")
        imgui.pop_item_width()
        imgui.spacing()

        any_in_progress = move_in_progress_flag[0] or grasp_in_progress_flag[0]

        if move_in_progress_flag[0]:
            imgui.push_style_color(imgui.COLOR_BUTTON,         0.70, 0.55, 0.00, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.85, 0.65, 0.00, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE,  0.90, 0.70, 0.00, 1.0)
            btn_label = " Moving…   "
        elif grasp_in_progress_flag[0]:
            imgui.push_style_color(imgui.COLOR_BUTTON,         0.70, 0.30, 0.00, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.85, 0.40, 0.00, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE,  0.90, 0.45, 0.00, 1.0)
            btn_label = " Grasping… "
        else:
            imgui.push_style_color(imgui.COLOR_BUTTON,         0.05, 0.55, 0.15, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.10, 0.70, 0.20, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE,  0.02, 0.45, 0.10, 1.0)
            btn_label = "  MOVE ▶   "

        if imgui.button(btn_label, width=win_w, height=28):
            if not any_in_progress:
                _current_obj_idx[0] = selected_object
                obj_world   = get_object_world_pos(sim.model, sim.data, selected_object)
                goal_xy     = (float(obj_world[0]), float(obj_world[1]))
                move_in_progress_flag[0] = True
                move_status = (f"Obj-{selected_object} → Slot-{selected_shelf}  "
                               f"src=({obj_world[0]:.2f},{obj_world[1]:.2f})")
                print(f"[MOVE] {move_status}")
                nav.navigate_to(goal_xy, on_complete=_on_nav_complete)

        imgui.pop_style_color(3)

        if any_in_progress:
            if move_in_progress_flag[0]:
                imgui.text_colored("● Navigating to object…", 1.0, 0.8, 0.0, 1.0)
            else:
                imgui.text_colored("● Grasping object…", 1.0, 0.5, 0.0, 1.0)
            if imgui.small_button("Cancel##mv"):
                nav.cancel()
                grasp_ctrl.cancel()
                move_in_progress_flag[0]  = False
                grasp_in_progress_flag[0] = False
                move_status = "Cancelled."
        elif grasp_ctrl.is_holding():
            held = grasp_ctrl.get_held_idx()
            imgui.text_colored(f"● Holding Obj-{held}", 0.3, 1.0, 0.5, 1.0)
            if imgui.small_button("Release##rel"):
                grasp_ctrl.cancel()
                move_status = "Object released."
        elif move_status:
            imgui.text_disabled(move_status[:80])

        imgui.end()
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(sim.window)

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()