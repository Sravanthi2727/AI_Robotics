#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
custom_envs.py
Custom gym environment classes for ETH NavRep:
  - CustomNavRepEnv: Extends NavRepTrainEnv with custom maps, polygonal obstacles,
    and advanced dynamic human agent models (groups, darting child, emergency gurney, surges).
  - CustomNavRepEncodedEnv: Wraps CustomNavRepEnv with world-model feature encoding (VAE_LSTM)
    for direct evaluation with pretrained NavRep PPO policies.
"""

from __future__ import print_function
import math
import numpy as np
import gym
from gym import spaces
from CMap2D import flatten_contours, CMap2D, CSimAgent
from pose2d import apply_tf_to_vel, inverse_pose2d, apply_tf_to_pose

from navrep.envs.navreptrainenv import NavRepTrainEnv
from navrep.envs.navreptrainencodedenv import NavRepTrainEncoder
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.action import ActionXYRot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.info import ReachGoal, Collision, CollisionOtherAgent, Timeout

from navrep_custom.scenarios import get_scenario_config, SCENARIO_REGISTRY


class CustomNavRepEnv(NavRepTrainEnv):
    """
    Subclass of NavRepTrainEnv supporting configurable custom scenarios,
    difficulty ladders, and dynamic crowd behaviors.
    """
    def __init__(self,
                 scenario="mallmate",
                 difficulty="MEDIUM",
                 silent=True,
                 lidar_legs=True,
                 collect_statistics=True):
        self.current_scenario_name = scenario
        self.current_difficulty = difficulty
        self.scenario_config = get_scenario_config(scenario)
        super(CustomNavRepEnv, self).__init__(
            scenario="test",
            silent=silent,
            adaptive=False,
            lidar_legs=lidar_legs,
            collect_statistics=collect_statistics
        )
        self.time_limit = self.scenario_config.time_limit_steps
        self.human_meta = []  # Metadata for dynamic behaviors (roles, waypoints, surges)

    def set_scenario(self, scenario_name, difficulty="MEDIUM"):
        """Switches the active scenario and difficulty configuration."""
        self.current_scenario_name = scenario_name
        self.current_difficulty = difficulty
        self.scenario_config = get_scenario_config(scenario_name)
        diff_cfg = self.scenario_config.get_difficulty_config(difficulty)
        self.time_limit = diff_cfg.get('time_limit', self.scenario_config.time_limit_steps)

    def _add_border_obstacle(self):
        """Robust border obstacle calculation supporting arbitrary polygon shapes."""
        all_agents = [self.soadrl_sim.robot] + self.soadrl_sim.humans + (self.soadrl_sim.other_robots or [])
        x_agents = [a.px for a in all_agents]
        y_agents = [a.py for a in all_agents]
        x_goals = [a.gx for a in all_agents]
        y_goals = [a.gy for a in all_agents]
        x_vertices = []
        y_vertices = []
        for poly in self.soadrl_sim.obstacle_vertices:
            for vx, vy in poly:
                x_vertices.append(vx)
                y_vertices.append(vy)
        BORDER_OFFSET = 1.0
        x_min = min(x_agents + x_goals + x_vertices) - BORDER_OFFSET
        x_max = max(x_agents + x_goals + x_vertices) + BORDER_OFFSET
        y_min = min(y_agents + y_goals + y_vertices) - BORDER_OFFSET
        y_max = max(y_agents + y_goals + y_vertices) + BORDER_OFFSET
        border_vertices = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)][::-1]
        self.border = [(x_min, x_max), (y_min, y_max)]
        self.soadrl_sim.obstacle_vertices.append(border_vertices)

    def reset(self, scenario_name=None, difficulty=None):
        if scenario_name is not None:
            self.set_scenario(scenario_name, difficulty or self.current_difficulty)

        cfg = self.scenario_config
        diff_cfg = cfg.get_difficulty_config(self.current_difficulty)

        # 1. Base reset of soadrl simulator to initialize internal buffers (map, time_step, etc.)
        self.soadrl_sim.reset('test', compute_local_map=False)
        self.steps_since_reset = 0
        self.episode_reward = 0.0
        self.soadrl_sim.phase = 'test'
        self.soadrl_sim.states = []
        self.soadrl_sim.attention_weights = []
        self.soadrl_sim.global_time = 0.0
        self.soadrl_sim.robot.time_step = self._get_dt()
        self.soadrl_sim.time_step = self._get_dt()

        # 2. Configure Robot Pose & Goal
        sx, sy, sth = cfg.robot_start
        gx, gy = cfg.robot_goal
        self.soadrl_sim.robot.set(sx, sy, gx, gy, 0.0, 0.0, sth)

        # 3. Configure Humans according to difficulty and role specifications
        speed_mult = diff_cfg.get('speed_mult', 1.0)
        humans = []
        self.human_meta = []

        # Base human agents
        for idx, h_spec in enumerate(cfg.base_humans):
            h = Human(self.soadrl_sim.config, 'humans')
            px, py, pth = h_spec['pos']
            hgx, hgy = h_spec['goal']
            rad = h_spec['radius']
            spd = h_spec['speed'] * speed_mult

            h.set(px, py, hgx, hgy, 0.0, 0.0, pth)
            h.radius = rad
            h.v_pref = spd
            h.time_step = self._get_dt()
            policy = ORCA()
            policy.configure(self.soadrl_sim.config)
            policy.time_step = self._get_dt()
            h.set_policy(policy)
            humans.append(h)
            self.human_meta.append({
                'role': h_spec.get('role', 'pedestrian'),
                'orig_goal': [hgx, hgy],
                'orig_start': [px, py],
                'base_speed': spd,
                'erratic_timer': 0
            })

        # Add extra difficulty agents if specified
        extra_count = diff_cfg.get('extra_humans', 0)
        bounds = cfg.world_bounds
        for e_i in range(extra_count):
            h = Human(self.soadrl_sim.config, 'humans')
            epx = np.random.uniform(bounds[0] + 2.0, bounds[1] - 2.0)
            epy = np.random.uniform(bounds[2] + 2.0, bounds[3] - 2.0)
            egx = np.random.uniform(bounds[0] + 2.0, bounds[1] - 2.0)
            egy = np.random.uniform(bounds[2] + 2.0, bounds[3] - 2.0)
            eth = np.random.uniform(-math.pi, math.pi)
            espd = np.random.uniform(0.6, 1.3) * speed_mult

            h.set(epx, epy, egx, egy, 0.0, 0.0, eth)
            h.radius = 0.3
            h.v_pref = espd
            h.time_step = self._get_dt()
            policy = ORCA()
            policy.configure(self.soadrl_sim.config)
            policy.time_step = self._get_dt()
            h.set_policy(policy)
            humans.append(h)
            self.human_meta.append({
                'role': 'extra_crowd',
                'orig_goal': [egx, egy],
                'orig_start': [epx, epy],
                'base_speed': espd,
                'erratic_timer': 0
            })

        self.soadrl_sim.humans = humans
        if self.soadrl_sim.other_robots is None:
            self.soadrl_sim.other_robots = []

        # 4. Configure Static Obstacles & Boundary
        self.soadrl_sim.obstacle_vertices = [list(poly) for poly in cfg.static_obstacles]
        self._add_border_obstacle()
        self.flat_contours = flatten_contours(self.soadrl_sim.obstacle_vertices)
        self.distances_travelled_in_base_frame = np.zeros((len(humans), 3))

        # Convert initial observation
        obs = self._convert_obs()
        return obs

    def step(self, action):
        self.steps_since_reset += 1
        self.total_steps += 1

        # Dynamic human behaviors based on scenario context
        for i, (human, meta) in enumerate(zip(self.soadrl_sim.humans, self.human_meta)):
            role = meta['role']
            # MallMate child erratic sudden direction / speed changes
            if 'child' in role:
                meta['erratic_timer'] += 1
                if meta['erratic_timer'] % 20 == 0:
                    # Child darts towards parent or nearby shop
                    jitter_x = np.random.uniform(-1.5, 1.5)
                    jitter_y = np.random.uniform(-1.5, 1.5)
                    human.gx = meta['orig_goal'][0] + jitter_x
                    human.gy = meta['orig_goal'][1] + jitter_y
                    human.v_pref = meta['base_speed'] * np.random.uniform(0.8, 1.6)

            # Railway surge commuter: rush across platform
            elif 'surge_commuter' in role:
                dist_to_goal = math.sqrt((human.px - human.gx)**2 + (human.py - human.gy)**2)
                if dist_to_goal < 0.8:
                    # Loop commuter flow
                    human.px, human.py = meta['orig_start']

            # MediRunner emergency gurney: keeps persistent aggressive forward path
            elif 'emergency_gurney' in role:
                dist_to_goal = math.sqrt((human.px - human.gx)**2 + (human.py - human.gy)**2)
                if dist_to_goal < 1.0:
                    human.px, human.py = meta['orig_start']

            # Oscillating flow for pedestrians reaching their waypoints
            elif 'pedestrian' in role or 'picker' in role:
                dist_to_goal = math.sqrt((human.px - human.gx)**2 + (human.py - human.gy)**2)
                if dist_to_goal < 0.6:
                    human.gx, human.gy = meta['orig_start']
                    meta['orig_start'] = [human.px, human.py]

        # Convert action to SOADRL coordinate system
        dr = action[2] * self._get_dt()
        vx_baselink = action[0] * np.cos(-dr) - action[1] * np.sin(-dr)
        vy_baselink = action[0] * np.sin(-dr) + action[1] * np.cos(-dr)
        soadrl_action = ActionXYRot(vx_baselink, vy_baselink, dr)

        _, _, reward, done, info = self.soadrl_sim.step(
            soadrl_action, compute_local_map=False, border=self.border
        )
        self._update_dist_travelled()
        obs = self._convert_obs()

        # Check time limit
        if not done and self.steps_since_reset >= self.time_limit:
            done = True
            info = Timeout()

        event = info
        return obs, reward, done, {'event': event}


class CustomNavRepEncodedEnv(gym.Env):
    """
    Wraps CustomNavRepEnv with NavRepTrainEncoder so that observation is
    an encoded vector (37,) compatible with pretrained NavRepCPolicy.
    """
    def __init__(self,
                 scenario="mallmate",
                 difficulty="MEDIUM",
                 backend="VAE_LSTM",
                 encoding="V_ONLY",
                 gpu=False,
                 silent=True):
        super(CustomNavRepEncodedEnv, self).__init__()
        self.raw_env = CustomNavRepEnv(
            scenario=scenario,
            difficulty=difficulty,
            silent=silent
        )
        self.encoder = NavRepTrainEncoder(backend, encoding, gpu=gpu)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = self.encoder.observation_space

    @property
    def soadrl_sim(self):
        return self.raw_env.soadrl_sim

    @property
    def current_scenario_name(self):
        return self.raw_env.current_scenario_name

    def set_scenario(self, scenario_name, difficulty="MEDIUM"):
        self.raw_env.set_scenario(scenario_name, difficulty)

    def _add_border_obstacle(self):
        """Robust border obstacle calculation supporting arbitrary polygon shapes."""
        all_agents = [self.soadrl_sim.robot] + self.soadrl_sim.humans + (self.soadrl_sim.other_robots or [])
        x_agents = [a.px for a in all_agents]
        y_agents = [a.py for a in all_agents]
        x_goals = [a.gx for a in all_agents]
        y_goals = [a.gy for a in all_agents]
        x_vertices = []
        y_vertices = []
        for poly in self.soadrl_sim.obstacle_vertices:
            for vx, vy in poly:
                x_vertices.append(vx)
                y_vertices.append(vy)
        BORDER_OFFSET = 1.0
        x_min = min(x_agents + x_goals + x_vertices) - BORDER_OFFSET
        x_max = max(x_agents + x_goals + x_vertices) + BORDER_OFFSET
        y_min = min(y_agents + y_goals + y_vertices) - BORDER_OFFSET
        y_max = max(y_agents + y_goals + y_vertices) + BORDER_OFFSET
        border_vertices = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)][::-1]
        self.border = [(x_min, x_max), (y_min, y_max)]
        self.soadrl_sim.obstacle_vertices.append(border_vertices)

    def reset(self, scenario_name=None, difficulty=None):
        self.encoder.reset()
        raw_obs = self.raw_env.reset(scenario_name, difficulty)
        encoded_obs = self.encoder._encode_obs(raw_obs, np.array([0.0, 0.0, 0.0]))
        return encoded_obs

    def step(self, action):
        # 2D velocity action [vx, vy, 0.0]
        full_action = np.array([action[0], action[1], 0.0], dtype=np.float32)
        raw_obs, reward, done, info = self.raw_env.step(full_action)
        encoded_obs = self.encoder._encode_obs(raw_obs, full_action)
        return encoded_obs, reward, done, info

    def close(self):
        self.raw_env.close()
