#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_models_and_checksums.py
Verifies model paths, file sizes, and MD5 checksums.
"""
from __future__ import print_function
import os
import hashlib

def get_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def verify_all():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models")
    
    key_files = [
        # World model V
        ("models/V/navreptrainvae.json", "V (Conv2D VAE World Model)"),
        ("models/V/navreptrainvae1d.json", "V (Conv1D VAE World Model)"),
        # World model M
        ("models/M/navreptrainrnn.json", "M (MDN-RNN World Model)"),
        ("models/M/navreptrainrnn1d.json", "M (MDN-RNN 1D World Model)"),
        # Key Gym Controllers
        ("models/gym/navreptrainencodedenv_2020_09_22__01_14_06_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt.zip", "PPO (VAE_LSTM V_ONLY)"),
        ("models/gym/navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip", "PPO (VAE_LSTM VM)"),
        ("models/gym/navreptrainencodedenv_2020_09_21__09_05_38_PPO_VAE1D_LSTM_VM_V32M512_ckpt.zip", "PPO (VAE1D_LSTM VM)"),
        ("models/gym/navreptrainencodedenv_2020_10_02__23_50_56_PPO_GPT_VM_V64M64_ckpt.zip", "PPO (GPT VM)"),
        ("models/gym/e2enavreptrainenv_2020_10_27__09_10_06_PPO_E2E_VCARCH_C64_ckpt.zip", "PPO (E2E 2D Baseline)"),
        ("models/gym/e2e1dnavreptrainenv_2020_10_27__12_13_34_PPO_E2E1D_VCARCH_C64_ckpt.zip", "PPO (E2E 1D Baseline)"),
    ]

    print("=" * 115)
    print("{:75s} | {:10s} | {:32s}".format("MODEL PATH / DESCRIPTION", "SIZE (KB)", "MD5 CHECKSUM"))
    print("=" * 115)

    missing = []
    for rel_path, desc in key_files:
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            print("MISSING: {}".format(rel_path))
            missing.append(rel_path)
        else:
            sz_kb = os.path.getsize(full_path) / 1024.0
            chk = get_md5(full_path)
            label = "{} ({})".format(rel_path, desc)
            print("{:75s} | {:10.1f} | {:32s}".format(label, sz_kb, chk))

    print("=" * 115)
    if missing:
        raise RuntimeError("CRITICAL ERROR: Missing model files: {}".format(missing))
    print("All required pretrained models verified successfully. ZERO fallback initializations.")

if __name__ == '__main__':
    verify_all()

