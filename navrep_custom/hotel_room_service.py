import math
import numpy as np


class HotelNavigationController(object):

    def __init__(self, max_speed=0.65):
        self.max_speed = max_speed
        self.waypoints = [
            [-7.0, 0.0],
            [-3.0, 0.0],
            [1.0, 0.0],
            [5.0, 0.0],
            [8.0, 0.0],
            [10.0, 0.0]
        ]
        self.current_wp_idx = 0
        self.stopped_steps = 0
        self.detour_y = 0.0

    def reset(self):
        self.current_wp_idx = 0
        self.stopped_steps = 0
        self.detour_y = 0.0

    def act(self, robot_pos, humans=None, action_dim=2):

        rx, ry = robot_pos[0], robot_pos[1]

        tx, ty = self.waypoints[self.current_wp_idx]
        ty += self.detour_y

        distance = math.hypot(tx - rx, ty - ry)

        while distance < 0.45 and self.current_wp_idx < len(self.waypoints) - 1:
            self.current_wp_idx += 1
            self.detour_y = 0.0
            tx, ty = self.waypoints[self.current_wp_idx]
            distance = math.hypot(tx - rx, ty - ry)

        if distance > 0.001:
            speed = min(self.max_speed, distance)
            vx = (tx - rx) / distance * speed
            vy = (ty - ry) / distance * speed
        else:
            vx, vy = 0.0, 0.0

        stopped = False

        if humans is not None:
            for h in humans:
                dx = h.px - rx
                dy = h.py - ry
                dist = math.hypot(dx, dy)

                safe_dist = 0.30 + h.radius + 0.25

                if dist < safe_dist:
                    stopped = True
                    break

                # Predict possible collision with moving pedestrians
                relative_vx = vx - h.vx
                relative_vy = vy - h.vy

                relative_speed = (
                    relative_vx * relative_vx +
                    relative_vy * relative_vy
                )

                if relative_speed > 0.001:
                    t = -(
                        dx * relative_vx +
                        dy * relative_vy
                    ) / relative_speed

                    if 0 < t < 1.5:
                        future_x = dx + relative_vx * t
                        future_y = dy + relative_vy * t

                        if math.hypot(future_x, future_y) < safe_dist:
                            stopped = True
                            break

        if stopped:
            vx, vy = 0.0, 0.0
            self.stopped_steps += 1

            if self.stopped_steps >= 8:
                self.detour_y = 0.8 if self.detour_y <= 0 else -0.8
                self.stopped_steps = 0
        else:
            self.stopped_steps = 0

        # Slow down near the final delivery point
        if self.current_wp_idx == len(self.waypoints) - 1:
            max_dock_speed = max(0.15, min(self.max_speed, distance * 0.8))
            total_speed = math.hypot(vx, vy)

            if total_speed > max_dock_speed:
                vx = vx / total_speed * max_dock_speed
                vy = vy / total_speed * max_dock_speed

        if action_dim == 2:
            return np.array([vx, vy], dtype=np.float32)

        return np.array([vx, vy, 0.0], dtype=np.float32)
