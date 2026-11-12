#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import imageio

from navrep_custom.custom_envs import CustomNavRepEnv
from navrep_custom.hotel_room_service import HotelNavigationController
from scripts_custom.record_scenario import render_scene_frame, verify_video_frames
from crowd_sim.envs.utils.info import ReachGoal, Collision, CollisionOtherAgent, Timeout


def record_hotel_demo():
    np.random.seed(42)

    output_path = "submission/task3_custom/videos/hotel_room_service_medium.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    env = CustomNavRepEnv(
        scenario="hotelroomservice",
        difficulty="MEDIUM",
        silent=True
    )

    controller = HotelNavigationController()
    controller.reset()

    fps = 12
    max_steps = 380
    dt = env._get_dt()
    total_frames = 0

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p"
    )

    try:
        env.reset("hotelroomservice", "MEDIUM")

        sim = env.soadrl_sim
        start_pos = [sim.robot.px, sim.robot.py]
        robot_trail = [start_pos[:]]

        status = "RUNNING"
        sim_time = 0.0
        step = 0
        done = False

        print("[HOTEL DEMO] Recording started...")

        while not done and step < max_steps:
            robot_pos = [sim.robot.px, sim.robot.py]

            action = controller.act(
                robot_pos,
                humans=sim.humans,
                action_dim=3
            )

            _, _, done, info = env.step(action)

            step += 1
            sim_time += dt

            robot_trail.append([sim.robot.px, sim.robot.py])

            event = info.get("event")

            if isinstance(event, ReachGoal):
                status = "GOAL REACHED"
                done = True
            elif isinstance(event, (Collision, CollisionOtherAgent)):
                status = "COLLISION"
                done = True
            elif isinstance(event, Timeout):
                status = "TIMEOUT"
                done = True

            frame = render_scene_frame(
                env=env,
                scenario_name="HOTEL ROOM SERVICE - MEDIUM",
                episode_num=1,
                step_num=step,
                sim_time=sim_time,
                status=status,
                robot_trail=robot_trail,
                start_pos=start_pos,
                policy_name="HOTEL NAVIGATION"
            )

            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            total_frames += 1

        if not done:
            status = "TIMEOUT"

        # Hold the final result on screen
        final_frame = render_scene_frame(
            env=env,
            scenario_name="HOTEL ROOM SERVICE - MEDIUM",
            episode_num=1,
            step_num=step,
            sim_time=sim_time,
            status=status,
            robot_trail=robot_trail,
            start_pos=start_pos,
            policy_name="HOTEL NAVIGATION"
        )

        final_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)

        for _ in range(int(fps * 2)):
            writer.append_data(final_rgb)
            total_frames += 1

        print("[HOTEL DEMO] Result: {}".format(status))
        print("[HOTEL DEMO] Steps: {}".format(step))
        print("[HOTEL DEMO] Final position: ({:.2f}, {:.2f})".format(
            sim.robot.px, sim.robot.py
        ))

    finally:
        writer.close()
        env.close()

    duration = float(total_frames) / fps

    print("[HOTEL DEMO] Video saved: {}".format(output_path))
    print("[HOTEL DEMO] Duration: {:.2f} seconds".format(duration))

    verify_video_frames(output_path)

    print("[HOTEL DEMO] Video quality check passed!")


if __name__ == "__main__":
    record_hotel_demo()
