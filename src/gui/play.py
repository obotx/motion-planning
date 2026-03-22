import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco
import glfw
import numpy as np
import imgui
from imgui.integrations.glfw import GlfwRenderer
from simulations.morph_i_free_move import ParallelRobot


class Joystick:
    def __init__(self, inner_radius=50, padding=20, ring_width=20, dead_zone=0.1):
        self.inner_radius = inner_radius
        self.padding = padding
        self.ring_width = ring_width
        self.outer_radius = inner_radius + padding + ring_width
        self.dead_zone = dead_zone

        self.xy_value = np.array([0.0, 0.0])
        self.yaw_drag_value = 0.0
        self.current_robot_yaw = 0.0

        self.is_active = False
        self._dragging = False
        self._mode = None

    def update_robot_yaw(self, yaw_rad):
        self.current_robot_yaw = float(yaw_rad)

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @property
    def value(self):
        if self._dragging and self._mode == 'yaw':
            command = self._normalize_angle(-self.yaw_drag_value)
            return self.xy_value, command
        return self.xy_value, None

    def draw(self, label="Joystick"):
        pos = imgui.get_cursor_screen_pos()
        center_x = pos.x + self.outer_radius
        center_y = pos.y + self.outer_radius

        draw_list = imgui.get_window_draw_list()

        draw_list.add_circle_filled(center_x, center_y, self.outer_radius, imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0))
        draw_list.add_circle_filled(center_x, center_y, self.inner_radius + self.padding, imgui.get_color_u32_rgba(0.1, 0.1, 0.1, 1.0))
        draw_list.add_circle_filled(center_x, center_y, self.inner_radius, imgui.get_color_u32_rgba(0.25, 0.25, 0.25, 1.0))

        io = imgui.get_io()
        mouse_x, mouse_y = io.mouse_pos
        mouse_down = io.mouse_down[0]

        if not self._dragging:
            dx0 = mouse_x - center_x
            dy0 = mouse_y - center_y
            dist0 = (dx0**2 + dy0**2)**0.5
            if dist0 <= self.outer_radius and mouse_down:
                if dist0 <= self.inner_radius:
                    self._mode = 'xy'
                    self._dragging = True
                elif dist0 >= self.inner_radius + self.padding:
                    self._mode = 'yaw'
                    self._dragging = True
                else:
                    self._mode = None

        dx = dy = 0.0
        if self._dragging and self._mode is not None:
            dx = mouse_x - center_x
            dy = mouse_y - center_y

            if self._mode == 'xy':
                dist = (dx**2 + dy**2)**0.5
                if dist > self.inner_radius:
                    dx = dx * self.inner_radius / dist
                    dy = dy * self.inner_radius / dist
                nx = dx / self.inner_radius
                ny = -dy / self.inner_radius
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
                self._mode = None
                self.xy_value = np.array([0.0, 0.0])
                self.is_active = False
        else:
            self.xy_value = np.array([0.0, 0.0])
            self.is_active = False

        knob_x = center_x + (dx if (self._mode == 'xy' and self._dragging) else 0)
        knob_y = center_y + (dy if (self._mode == 'xy' and self._dragging) else 0)
        draw_list.add_circle_filled(
            knob_x, knob_y, 8,
            imgui.get_color_u32_rgba(0.0, 0.8, 1.0, 1.0 if (self._mode == 'xy' and self._dragging) else 0.6)
        )

        mirrored_yaw = self._normalize_angle(-self.current_robot_yaw)
        ring_radius = self.inner_radius + self.padding + self.ring_width / 2
        indicator_x = center_x + ring_radius * np.cos(mirrored_yaw)
        indicator_y = center_y + ring_radius * np.sin(mirrored_yaw)
        draw_list.add_circle_filled(
            indicator_x, indicator_y, 6,
            imgui.get_color_u32_rgba(1.0, 0.7, 0.0, 1.0)
        )

        imgui.invisible_button(label, self.outer_radius * 2, self.outer_radius * 2)
        return self.xy_value, self._normalize_angle(-self.yaw_drag_value) if (self._dragging and self._mode == 'yaw') else None


def _map_pct_to_value(pct, low, high):
    return low + (high - low) * (pct / 100.0)

def _map_value_to_pct(val, low, high):
    return (val - low) / (high - low) * 100.0


def main():
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.DEPTH_BITS, 24)

    xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'env', 'market_world_plain.xml'))
    sim = ParallelRobot(xml_path, run_mode="glfw", record=False)

    glfw.make_context_current(sim.window)
    glfw.swap_interval(1)  

    imgui.create_context()
    impl = GlfwRenderer(sim.window, attach_callbacks=False)

    initial_x, initial_y, initial_yaw = sim.localization()
    target_x, target_y, target_yaw = initial_x, initial_y, initial_yaw

    joystick = Joystick(inner_radius=50, padding=20, ring_width=20, dead_zone=0.1)
    joystick.update_robot_yaw(initial_yaw)

    paused = False
    dt = sim.model.opt.timestep

    sim.camera.distance = 5.0
    sim.camera.azimuth = 90
    sim.camera.elevation = -45
    sim.camera.lookat[:] = [0, 0, 0]

    xpos, ypos = glfw.get_cursor_pos(sim.window)
    sim._last_mouse_x = xpos
    sim._last_mouse_y = ypos

    while not glfw.window_should_close(sim.window):
        # sim.get_keyframe("home")
        glfw.poll_events()
        impl.process_inputs()


        io = imgui.get_io()
        if not io.want_capture_mouse:
            xpos, ypos = glfw.get_cursor_pos(sim.window)
            dx = xpos - sim._last_mouse_x
            dy = ypos - sim._last_mouse_y
            sim._last_mouse_x, sim._last_mouse_y = xpos, ypos

            left = glfw.get_mouse_button(sim.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
            right = glfw.get_mouse_button(sim.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
            middle = glfw.get_mouse_button(sim.window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS

            factor = 0.001
            if left:
                mujoco.mjv_moveCamera(sim.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, dx*factor, dy*factor, sim.scene, sim.camera)
            elif right:
                mujoco.mjv_moveCamera(sim.model, mujoco.mjtMouse.mjMOUSE_MOVE_H, dx*factor, dy*factor, sim.scene, sim.camera)
            elif middle:
                mujoco.mjv_moveCamera(sim.model, mujoco.mjtMouse.mjMOUSE_ZOOM, dx*factor, dy*factor, sim.scene, sim.camera)
        else:
            xpos, ypos = glfw.get_cursor_pos(sim.window)
            sim._last_mouse_x = xpos
            sim._last_mouse_y = ypos

        actual_x, actual_y, actual_yaw = sim.localization()
        joystick.update_robot_yaw(actual_yaw)  
        
        # SIMULATION STEP
        if not paused:
            (x_local, y_local), yaw_command = joystick.value

            if yaw_command is not None:
                target_yaw = yaw_command

            cos_a = np.cos(actual_yaw)
            sin_a = np.sin(actual_yaw)

            world_vx =  cos_a * y_local + sin_a * x_local
            world_vy =  sin_a * y_local - cos_a * x_local  

            speed = 1.0  # m/s
            target_x += world_vx * speed * dt
            target_y += world_vy * speed * dt

            with sim._target_lock:
                sim.target_base = np.array([target_x, target_y, target_yaw])

            sim.step_simulation(render=False)

        # RENDERING
        fb_width, fb_height = glfw.get_framebuffer_size(sim.window)
        sim.viewport.width = fb_width
        sim.viewport.height = fb_height

        mujoco.mjv_updateScene(sim.model, sim.data, sim.opt, None, sim.camera, mujoco.mjtCatBit.mjCAT_ALL, sim.scene)
        mujoco.mjr_render(sim.viewport, sim.scene, sim.ctx)

        imgui.new_frame()
        imgui.set_next_window_position(10, 10)
        imgui.set_next_window_bg_alpha(0.95)

        flags = (
            imgui.WINDOW_NO_TITLE_BAR |
            imgui.WINDOW_NO_RESIZE |
            imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_COLLAPSE |
            imgui.WINDOW_ALWAYS_AUTO_RESIZE
        )

        imgui.begin("Robot Control", flags=flags)
        
        ik_enabled = sim.use_ik
        clicked, ik_enabled = imgui.checkbox("IK Control", ik_enabled)
        if clicked:
            with sim._target_lock:
                was_ik = sim.use_ik
                sim.use_ik = ik_enabled

                if ik_enabled:
                    l_enc, r_enc = sim.get_encoder()
                    fk_left = sim.fk(*l_enc)
                    fk_right = sim.fk(*r_enc)
                    sim.target_left = np.array(fk_left) if fk_left is not None else np.array([0.5, 0.0, 0.5])
                    sim.target_right = np.array(fk_right) if fk_right is not None else np.array([-0.5, 0.0, 0.5])
                else:
                    l_enc, r_enc = sim.get_encoder()
                    sim.direct_arm_commands[0:4] = l_enc  
                    sim.direct_arm_commands[4:8] = r_enc

        # ARM CONNTROLS
        imgui.separator()
        imgui.text("ARM1")
        if ik_enabled:
            imgui.push_item_width(180)
            tl = list(sim.target_left)
            _, x = imgui.slider_float("X##ARM1", tl[0], -1.0, 1.0, "%.3f")
            _, y = imgui.slider_float("Y##ARM1", tl[1], -1.0, 1.0, "%.3f")
            _, z = imgui.slider_float("Z##ARM1", tl[2], -1.0, 1.0, "%.3f")
            imgui.pop_item_width()
            if x != tl[0] or y != tl[1] or z != tl[2]:
                with sim._target_lock:
                    sim.target_left = np.array([x, y, z])
        else:
            imgui.push_item_width(180)
            h1_int, h2_int, a1_int, th = sim.direct_arm_commands[0:4]

            h1_ui = np.interp(h1_int, [0.0, 1.5], [-75.0, 75.0])
            h2_ui = np.interp(h2_int, [0.0, 1.5], [-75.0, 75.0])
            a1_ui = np.interp(a1_int, [0.0, 0.7], [-45.0, 45.0])

            _, h1_ui = imgui.slider_float("H1##ARM1", h1_ui, -75.0, 75.0, "%.1f°")
            _, h2_ui = imgui.slider_float("H2##ARM1", h2_ui, -75.0, 75.0, "%.1f°")
            _, a1_ui = imgui.slider_float("A1##ARM1", a1_ui, -45.0, 45.0, "%.1f°")
            _, th_deg = imgui.slider_float("θ##ARM1", np.degrees(th), np.degrees(-np.pi/2), np.degrees(np.pi/2), "%.1f°")
            th_deg = np.clip(th_deg, np.degrees(-np.pi), np.degrees(np.pi))

            h1_out = np.interp(h1_ui, [-75.0, 75.0], [0.0, 1.5])
            h2_out = np.interp(h2_ui, [-75.0, 75.0], [0.0, 1.5])
            a1_out = np.interp(a1_ui, [-45.0, 45.0], [0.0, 0.7])

            imgui.pop_item_width()
            with sim._target_lock:
                sim.direct_arm_commands[0:4] = [h1_out, h2_out, a1_out, np.radians(th_deg)]

        imgui.separator()

        imgui.text("ARM2")
        if ik_enabled:
            imgui.push_item_width(180)
            tr = list(sim.target_right)
            _, x = imgui.slider_float("X##ARM2", tr[0], -1.0, 1.0, "%.3f")
            _, y = imgui.slider_float("Y##ARM2", tr[1], -1.0, 1.0, "%.3f")
            _, z = imgui.slider_float("Z##ARM2", tr[2], -1.0, 1.0, "%.3f")
            imgui.pop_item_width()
            if x != tr[0] or y != tr[1] or z != tr[2]:
                with sim._target_lock:
                    sim.target_right = np.array([x, y, z])
        else:
            imgui.push_item_width(180)
            h1_int, h2_int, a1_int, th = sim.direct_arm_commands[4:8]

            h1_ui = np.interp(h1_int, [0.0, 1.5], [-75.0, 75.0])
            h2_ui = np.interp(h2_int, [0.0, 1.5], [-75.0, 75.0])
            a1_ui = np.interp(a1_int, [0.0, 0.7], [-45.0, 45.0])

            _, h1_ui = imgui.slider_float("H1##ARM2", h1_ui, -75.0, 75.0, "%.1f°")
            _, h2_ui = imgui.slider_float("H2##ARM2", h2_ui, -75.0, 75.0, "%.1f°")
            _, a1_ui = imgui.slider_float("A1##ARM2", a1_ui, -45.0, 45.0, "%.1f°")
            _, th_deg = imgui.slider_float("θ##ARM2", np.degrees(th), np.degrees(-np.pi/2), np.degrees(np.pi/2), "%.1f°")
            th_deg = np.clip(th_deg, np.degrees(-np.pi), np.degrees(np.pi))

            h1_out = np.interp(h1_ui, [-75.0, 75.0], [0.0, 1.5])
            h2_out = np.interp(h2_ui, [-75.0, 75.0], [0.0, 1.5])
            a1_out = np.interp(a1_ui, [-45.0, 45.0], [0.0, 0.7])

            imgui.pop_item_width()
            with sim._target_lock:
                sim.direct_arm_commands[4:8] = [h1_out, h2_out, a1_out, np.radians(th_deg)]

        # GRIPPERS
        imgui.separator()
        imgui.text("Grippers")
        imgui.text_disabled("(0% = Open, 100% = Closed)")

        GRIPPER_OPEN_POS = -1.0
        GRIPPER_CLOSED_POS = 0.8

        imgui.push_item_width(180)
        if len(sim.gripper_ids_left) >= 7: 
            current_val = sim.data.ctrl[sim.gripper_ids_left[0]]
            left_pct = (current_val - GRIPPER_OPEN_POS) / (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS) * 100.0
            left_pct = np.clip(left_pct, 0.0, 100.0)

            changed, left_pct = imgui.slider_float("Left##GRIP", left_pct, 0.0, 100.0, "%.0f%%")
            if changed:
                target_val = GRIPPER_OPEN_POS + (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS) * (left_pct / 100.0)
                for idx in [0, 3, 6]:
                    sim.data.ctrl[sim.gripper_ids_left[idx]] = target_val
        else:
            imgui.text("Left gripper not available")
        imgui.pop_item_width()

        imgui.push_item_width(180)
        if len(sim.gripper_ids_right) >= 7:
            current_val = sim.data.ctrl[sim.gripper_ids_right[0]]
            right_pct = (current_val - GRIPPER_OPEN_POS) / (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS) * 100.0
            right_pct = np.clip(right_pct, 0.0, 100.0)

            changed, right_pct = imgui.slider_float("Right##GRIP", right_pct, 0.0, 100.0, "%.0f%%")
            if changed:
                target_val = GRIPPER_OPEN_POS + (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS) * (right_pct / 100.0)
                for idx in [0, 3, 6]:
                    sim.data.ctrl[sim.gripper_ids_right[idx]] = target_val
        else:
            imgui.text("Right gripper not available")
        imgui.pop_item_width()

        # WRISTS
        imgui.separator()
        
        WRIST_X_RANGE = (-0.8, 0.8)
        WRIST_Y_RANGE = (-0.8, 0.8)
        WRIST_Z_RANGE = (-np.pi, np.pi)

        imgui.text("Left Wrist")
        imgui.text_disabled("(0% = Min, 100% = Max)")
        imgui.push_item_width(180)
        if len(sim.gripper_ids_left) > 13:
            wx_val = sim.data.ctrl[sim.gripper_ids_left[11]]
            wx_pct = _map_value_to_pct(wx_val, *WRIST_X_RANGE)
            wx_pct = np.clip(wx_pct, 0, 100)
            changed, wx_pct = imgui.slider_float("X##LW", wx_pct, 0.0, 100.0, "%.0f%%")
            if changed:
                sim.data.ctrl[sim.gripper_ids_left[11]] = _map_pct_to_value(wx_pct, *WRIST_X_RANGE)

            wy_val = sim.data.ctrl[sim.gripper_ids_left[12]]
            wy_pct = _map_value_to_pct(wy_val, *WRIST_Y_RANGE)
            wy_pct = np.clip(wy_pct, 0, 100)
            changed, wy_pct = imgui.slider_float("Y##LW", wy_pct, 0.0, 100.0, "%.0f%%")
            if changed:
                sim.data.ctrl[sim.gripper_ids_left[12]] = _map_pct_to_value(wy_pct, *WRIST_Y_RANGE)

            wz_val = sim.data.ctrl[sim.gripper_ids_left[13]]
            wz_pct = _map_value_to_pct(wz_val, *WRIST_Z_RANGE)
            wz_pct = np.clip(wz_pct, 0, 100)
            changed, wz_pct = imgui.slider_float("Z##LW", wz_pct, 0.0, 100.0, "%.0f%%")
            if changed:
                sim.data.ctrl[sim.gripper_ids_left[13]] = _map_pct_to_value(wz_pct, *WRIST_Z_RANGE)
        else:
            imgui.text("Left wrist not available")
        imgui.pop_item_width()

        imgui.text("Right Wrist")
        imgui.text_disabled("(0% = Min, 100% = Max)")
        imgui.push_item_width(180)
        if len(sim.gripper_ids_right) > 13:
            wx_val = sim.data.ctrl[sim.gripper_ids_right[11]]
            wx_pct = _map_value_to_pct(wx_val, *WRIST_X_RANGE)
            wx_pct = np.clip(wx_pct, 0, 100)
            changed, wx_pct = imgui.slider_float("X##RW", wx_pct, 0.0, 100.0, "%.0f%%")
            if changed:
                sim.data.ctrl[sim.gripper_ids_right[11]] = _map_pct_to_value(wx_pct, *WRIST_X_RANGE)

            wy_val = sim.data.ctrl[sim.gripper_ids_right[12]]
            wy_pct = _map_value_to_pct(wy_val, *WRIST_Y_RANGE)
            wy_pct = np.clip(wy_pct, 0, 100)
            changed, wy_pct = imgui.slider_float("Y##RW", wy_pct, 0.0, 100.0, "%.0f%%")
            if changed:
                sim.data.ctrl[sim.gripper_ids_right[12]] = _map_pct_to_value(wy_pct, *WRIST_Y_RANGE)

            wz_val = sim.data.ctrl[sim.gripper_ids_right[13]]
            wz_pct = _map_value_to_pct(wz_val, *WRIST_Z_RANGE)
            wz_pct = np.clip(wz_pct, 0, 100)
            changed, wz_pct = imgui.slider_float("Z##RW", wz_pct, 0.0, 100.0, "%.0f%%")
            if changed:
                sim.data.ctrl[sim.gripper_ids_right[13]] = _map_pct_to_value(wz_pct, *WRIST_Z_RANGE)
        else:
            imgui.text("Right wrist not available")
        imgui.pop_item_width()
        
        # HAND BEARINGS
        imgui.separator()
        imgui.text("Hand Bearings")
        imgui.text_disabled("(0% = -90°, 100% = +90°)")

        BEARING_RANGE = (-1.57, 1.57)  # ≈ (-π/2, π/2)
        
        imgui.push_item_width(180)
        if len(sim.gripper_ids_left) > 14:
            bearing_val = sim.data.ctrl[sim.gripper_ids_left[14]]
            bearing_pct = _map_value_to_pct(bearing_val, *BEARING_RANGE)
            bearing_pct = np.clip(bearing_pct, 0, 100)
            changed, bearing_pct = imgui.slider_float("Left##HB", bearing_pct, 0.0, 100.0, "%.0f%%")
            if changed:
                sim.data.ctrl[sim.gripper_ids_left[14]] = _map_pct_to_value(bearing_pct, *BEARING_RANGE)
        else:
            imgui.text("Left bearing not available")
        imgui.pop_item_width()

        imgui.push_item_width(180)
        if len(sim.gripper_ids_right) > 14:
            bearing_val = sim.data.ctrl[sim.gripper_ids_right[14]]
            bearing_pct = _map_value_to_pct(bearing_val, *BEARING_RANGE)
            bearing_pct = np.clip(bearing_pct, 0, 100)
            changed, bearing_pct = imgui.slider_float("Right##HB", bearing_pct, 0.0, 100.0, "%.0f%%")
            if changed:
                sim.data.ctrl[sim.gripper_ids_right[14]] = _map_pct_to_value(bearing_pct, *BEARING_RANGE)
        else:
            imgui.text("Right bearing not available")
        imgui.pop_item_width()
        
        imgui.separator()
        _, paused = imgui.checkbox("Pause", paused)
        if imgui.button("Reset Robot"):
            sim.reset("home")
            rx, ry, ryaw = sim.localization()
            target_x, target_y, target_yaw = rx, ry, ryaw
            joystick.update_robot_yaw(ryaw)
            with sim._target_lock:
                sim.target_base = np.array([target_x, target_y, target_yaw])
                sim.target_left = sim.fk(*sim.get_encoder()[0]) 
                sim.target_right = sim.fk(*sim.get_encoder()[1])

        imgui.separator()
        imgui.text("Joystick:")
        joystick.draw("MainJoystick")
        imgui.text(f"[X: {target_x:.1f}, Y: {target_y:.1f}, Yaw: {np.degrees(target_yaw):.1f}°]")
        imgui.end()

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(sim.window)

    impl.shutdown()


if __name__ == "__main__":
    main()