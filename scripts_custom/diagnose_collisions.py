#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_collisions.py
Step 2.3: Detailed collision investigation for all existing scenarios.
Logs spawn clearance, step of collision, obstacle type (static vs dynamic), and action values.
"""
from __future__ import print_function
import os
import sys
import numpy as np
from stable_baselines import PPO2
from navrep.envs.navreptrainencodedenv import NavRepTrainEncoder, NavRepTrainEncodedEnv
from navrep.envs.encodedenv import EncodedEnv
from navrep.scripts.test_navrep import NavRepCPolicy
from crowd_sim.envs.utils.info import ReachGoal, Collision, CollisionOtherAgent, Timeout

class ScenarioEncodedEnv(EncodedEnv):
    def reset_to_scenario(self, scenario_name=None):
        self.encoder.reset()
        obs = super(EncodedEnv, self).reset(set_scenario=scenario_name)
        h = self.encoder._encode_obs(obs, np.array([0., 0., 0.]))
        return h

def run_diagnostics():
    model_path = "/home/sravanthi/Desktop/navrep/models/gym/navreptrainencodedenv_2020_09_22__01_14_06_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt.zip"
    print("Loading model: {}".format(model_path))
    model = PPO2.load(model_path)
    policy = NavRepCPolicy(model)
    encoder = NavRepTrainEncoder("VAE_LSTM", "V_ONLY", gpu=False)

    scenarios = [
        "navreptrain_train",
        "navreptrain_test",
        "irosasl1",
        "irosasl2",
        "irosasl3",
        "irosasl_office_j1",
        "irosasl_office_j2",
        "irosasl_office_j3",
        "irosunity_scene_map1",
        "irosunity_scene_map2",
        "irosunity_scene_map3"
    ]

    print("\n" + "="*80)
    print("{:22s} | {:7s} | {:9s} | {:8s} | {:10s} | {}".format(
        "SCENARIO", "OUTCOME", "COLL_STEP", "DAMAGE_S", "DAMAGE_H", "SPAWN CLEARANCE (STATIC / HUMAN)"
    ))
    print("="*80)

    for sc in scenarios:
        is_navreptrain = sc.startswith("navreptrain")
        if is_navreptrain:
            sub = sc.replace("navreptrain_", "")
            env = NavRepTrainEncodedEnv("VAE_LSTM", "V_ONLY", scenario=sub, silent=True, gpu=False)
        else:
            env = ScenarioEncodedEnv(None, None, silent=True, encoder=encoder, collect_trajectories=False)

        # Run 3 sample episodes
        for ep in range(3):
            np.random.seed(42 + ep * 10)
            if is_navreptrain:
                ob = env.reset()
                sim = env.soadrl_sim
                rx, ry = sim.robot.px, sim.robot.py
                # clearance to humans
                h_dists = [np.hypot(h.px - rx, h.py - ry) - h.radius for h in sim.humans]
                min_h = min(h_dists) if h_dists else 999.0
                min_s = 2.0 # circular arena default
            else:
                ob = env.reset_to_scenario(sc)
                rlenv = env.iarlenv.rlenv
                rx, ry = rlenv.virtual_peppers[0].pos[:2]
                h_dists = [np.hypot(vp.pos[0] - rx, vp.pos[1] - ry) - rlenv.vp_radii[i+1]
                           for i, vp in enumerate(rlenv.virtual_peppers[1:])]
                min_h = min(h_dists) if h_dists else 999.0
                # Static clearance via lidar min range at spawn
                min_s = float(np.min(ob[:1080])) if len(ob) > 1080 else 1.0

            done = False
            step = 0
            outcome = "TIMEOUT"
            coll_step = -1
            dmg_static = 0.0
            dmg_human = 0.0

            while not done and step < 250:
                act = policy.act(ob)
                ob, rew, done, info = env.step(act)
                step += 1
                if is_navreptrain:
                    ev = info.get('event', None)
                    if isinstance(ev, ReachGoal):
                        outcome = "SUCCESS"
                        done = True
                    elif isinstance(ev, (Collision, CollisionOtherAgent)):
                        outcome = "COLLISION"
                        coll_step = step
                        done = True
                else:
                    if info.get('goal_reached', False):
                        outcome = "SUCCESS"
                        done = True
                    elif env.iarlenv.rlenv.episode_damage[0] >= env.max_episode_damage:
                        outcome = "COLLISION"
                        coll_step = step
                        dmg_static = env.iarlenv.rlenv.episode_damage_from_static[0]
                        dmg_human = env.iarlenv.rlenv.episode_damage[0] - dmg_static
                        done = True
                    elif env.iarlenv.rlenv.episode_step[0] >= env.max_episode_length:
                        outcome = "TIMEOUT"
                        done = True

            print("{:22s} | {:7s} | step {:4d} | {:8.2f} | {:10.2f} | static={:.2f}m, human={:.2f}m".format(
                sc if ep == 0 else "  ep {}".format(ep),
                outcome, coll_step, dmg_static, dmg_human, min_s, min_h
            ))

        env.close()

if __name__ == '__main__':
    run_diagnostics()

