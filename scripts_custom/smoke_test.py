#!/usr/bin/env python
import os
import sys

# Ensure matplotlib uses a writable temp cache directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import gym
import navrep
from navrep.envs.navreptrainenv import NavRepTrainEnv
from navrep.envs.ianenv import IANEnv

def run_smoke_test():
    print("=== NavRep Smoke Test ===")
    print("Python version:", sys.version)
    print("NavRep package path:", navrep.__file__)

    # Test 1: NavRepTrainEnv
    print("\n[1/2] Initializing NavRepTrainEnv(silent=True)...")
    env = NavRepTrainEnv(silent=True)
    obs = env.reset()
    print("Reset successful. Observation shape:", [o.shape for o in obs])

    print("Running 50 random action steps...")
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    env.close()
    print("NavRepTrainEnv completed 50 steps successfully.")

    # Test 2: IANEnv
    print("\n[2/2] Initializing IANEnv(silent=True)...")
    ian_env = IANEnv(silent=True)
    obs = ian_env.reset()
    print("Reset successful. Observation shape:", [o.shape for o in obs])

    print("Running 50 random action steps...")
    for step in range(50):
        action = ian_env.action_space.sample()
        obs, reward, done, info = ian_env.step(action)
        if done:
            obs = ian_env.reset()
    ian_env.close()
    print("IANEnv completed 50 steps successfully.")

    print("\n=== Smoke test PASSED cleanly! ===")

if __name__ == "__main__":
    run_smoke_test()

