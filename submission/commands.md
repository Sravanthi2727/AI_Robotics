# Reproduction Commands Log

This file records the exact reproducible commands for all tasks, benchmarks, tests, and scenario recordings.

## Phase 1 / Task 1: Environment and Smoke Testing

### 1. Minimal Smoke Test
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib /home/sravanthi/Desktop/navrep/env/bin/python scripts_custom/smoke_test.py
```

### 2. Pretrained Policy Cross-Testing in IANEnv
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib /home/sravanthi/Desktop/navrep/env/bin/python -m navrep.scripts.cross_test_navreptrain_in_ianenv --backend VAE_LSTM --encoding V_ONLY --no-gpu --n 1
```

