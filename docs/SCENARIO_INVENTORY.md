# NavRep Scenario and Environment Inventory

This document enumerates every scenario, environment class, launch procedure, required external assets, and operational status in the `ethz-asl/navrep` repository.

## Inventory Summary Table

| Scenario Name | Environment Class | Launch Command / API | Needs Extra Assets? | Operational Status | Notes |
|---|---|---|---|---|---|
| `navreptrain_train` | `NavRepTrainEnv` | `python -m navrep.scripts.play_navreptrainenv` | None (Procedural SOADRL crowd) | **VERIFIED OPERATIONAL** | Procedural training environment with dynamic obstacles & ORCA humans |
| `navreptrain_val` | `NavRepTrainEnv` | `NavRepTrainEnv(scenario='val')` | None (Procedural) | **VERIFIED OPERATIONAL** | Fixed seed validation set |
| `navreptrain_test` | `NavRepTrainEnv` | `NavRepTrainEnv(scenario='test')` | None (Procedural) | **VERIFIED OPERATIONAL** | Standard evaluation environment for SOADRL benchmark |
| `irosasl1` | `IANEnv` | `python -m navrep.scripts.cross_test_navreptrain_in_ianenv --scenario irosasl1` | `asl_pepper_2d_sim_maps` (included) | **VERIFIED OPERATIONAL** | ETH ASL floor map, 1 human obstacle |
| `irosasl2` | `IANEnv` | `IANEnv().reset('irosasl2')` | `asl_pepper_2d_sim_maps` (included) | **VERIFIED OPERATIONAL** | ETH ASL floor map, 2 human obstacles |
| `irosasl3` | `IANEnv` | `IANEnv().reset('irosasl3')` | `asl_pepper_2d_sim_maps` (included) | **VERIFIED OPERATIONAL** | ETH ASL floor map, 3 human obstacles |
| `irosasl4` | `IANEnv` | `IANEnv().reset('irosasl4')` | `asl_pepper_2d_sim_maps` (included) | **VERIFIED OPERATIONAL** | ETH ASL floor map, 4 human obstacles |
| `irosasl5` | `IANEnv` | `IANEnv().reset('irosasl5')` | `asl_pepper_2d_sim_maps` (included) | **VERIFIED OPERATIONAL** | ETH ASL floor map, 5 human obstacles |
| `irosasl6` | `IANEnv` | `IANEnv().reset('irosasl6')` | `asl_pepper_2d_sim_maps` (included) | **VERIFIED OPERATIONAL** | ETH ASL floor map, 6 human obstacles |
| `irosasl_office_j1` | `IANEnv` | `IANEnv().reset('irosasl_office_j1')` | `asl_office_j` map (included) | **VERIFIED OPERATIONAL** | Complex office hallway map J, 1 human |
| `irosasl_office_j2` | `IANEnv` | `IANEnv().reset('irosasl_office_j2')` | `asl_office_j` map (included) | **VERIFIED OPERATIONAL** | Complex office hallway map J, 2 humans |
| `irosasl_office_j3` | `IANEnv` | `IANEnv().reset('irosasl_office_j3')` | `asl_office_j` map (included) | **VERIFIED OPERATIONAL** | Complex office hallway map J, 3 humans |
| `irosasl_office_j4` | `IANEnv` | `IANEnv().reset('irosasl_office_j4')` | `asl_office_j` map (included) | **VERIFIED OPERATIONAL** | Complex office hallway map J, 4 humans |
| `irosasl_office_j5` | `IANEnv` | `IANEnv().reset('irosasl_office_j5')` | `asl_office_j` map (included) | **VERIFIED OPERATIONAL** | Complex office hallway map J, 5 humans |
| `irosasl_office_j6` | `IANEnv` | `IANEnv().reset('irosasl_office_j6')` | `asl_office_j` map (included) | **VERIFIED OPERATIONAL** | Complex office hallway map J, 6 humans |
| `irosunity_scene_map1` | `IANEnv` | `IANEnv().reset('irosunity_scene_map1')` | `unity_scene_map` (included) | **VERIFIED OPERATIONAL** | 2D projection of Unity scene, 1 human |
| `irosunity_scene_map2` | `IANEnv` | `IANEnv().reset('irosunity_scene_map2')` | `unity_scene_map` (included) | **VERIFIED OPERATIONAL** | 2D projection of Unity scene, 2 humans |
| `irosunity_scene_map3` | `IANEnv` | `IANEnv().reset('irosunity_scene_map3')` | `unity_scene_map` (included) | **VERIFIED OPERATIONAL** | 2D projection of Unity scene, 3 humans |
| `irosunity_scene_map4` | `IANEnv` | `IANEnv().reset('irosunity_scene_map4')` | `unity_scene_map` (included) | **VERIFIED OPERATIONAL** | 2D projection of Unity scene, 4 humans |
| `irosunity_scene_map5` | `IANEnv` | `IANEnv().reset('irosunity_scene_map5')` | `unity_scene_map` (included) | **VERIFIED OPERATIONAL** | 2D projection of Unity scene, 5 humans |
| `irosunity_scene_map6` | `IANEnv` | `IANEnv().reset('irosunity_scene_map6')` | `unity_scene_map` (included) | **VERIFIED OPERATIONAL** | 2D projection of Unity scene, 6 humans |
| `rlasl1` | `IANEnv` | `IANEnv().reset('rlasl1')` | `asl` map (included) | **VERIFIED OPERATIONAL** | Reinforcement learning curriculum setup 1 |
| `rlasl2` | `IANEnv` | `IANEnv().reset('rlasl2')` | `asl` map (included) | **VERIFIED OPERATIONAL** | Reinforcement learning curriculum setup 2 |
| `rlasl3` | `IANEnv` | `IANEnv().reset('rlasl3')` | `asl` map (included) | **VERIFIED OPERATIONAL** | Reinforcement learning curriculum setup 3 |
| `rlasl_office_j1` | `IANEnv` | `IANEnv().reset('rlasl_office_j1')` | `asl_office_j` map (included) | **VERIFIED OPERATIONAL** | RL curriculum office layout 1 |
| `rlasl_office_j2` | `IANEnv` | `IANEnv().reset('rlasl_office_j2')` | `asl_office_j` map (included) | **VERIFIED OPERATIONAL** | RL curriculum office layout 2 |
| `rlasl_office_j3` | `IANEnv` | `IANEnv().reset('rlasl_office_j3')` | `asl_office_j` map (included) | **VERIFIED OPERATIONAL** | RL curriculum office layout 3 |
| `rlunity_scene_map1` | `IANEnv` | `IANEnv().reset('rlunity_scene_map1')` | `unity_scene_map` (included) | **VERIFIED OPERATIONAL** | RL curriculum unity scene layout 1 |
| `rlunity_scene_map2` | `IANEnv` | `IANEnv().reset('rlunity_scene_map2')` | `unity_scene_map` (included) | **VERIFIED OPERATIONAL** | RL curriculum unity scene layout 2 |
| `rlunity_scene_map3` | `IANEnv` | `IANEnv().reset('rlunity_scene_map3')` | `unity_scene_map` (included) | **VERIFIED OPERATIONAL** | RL curriculum unity scene layout 3 |
| `marksimple` | `MarkEnv` | `MarkEnv(maps=['marksimple'])` | `marksimple` map (included) | **VERIFIED OPERATIONAL** | Simple static obstacles |
| `markcomplex` | `MarkEnv` | `MarkEnv(maps=['markcomplex'])` | `markcomplex` map (included) | **VERIFIED OPERATIONAL** | Complex corridor obstacles |
| `markclutter` | `MarkEnv` | `MarkEnv(maps=['markclutter'])` | `markclutter` map (included) | **VERIFIED OPERATIONAL** | High static clutter layout |
| `markmaze` | `MarkEnv` | `MarkEnv(maps=['markmaze'])` | `markmaze` map (included) | **VERIFIED OPERATIONAL** | Labyrinth maze layout |
| `ETH Unity 3D Binary` | Unity Linux Executable | `./ETH_Unity_Sim.x86_64` | Proprietary standalone Unity binary | **ASSET UNAVAILABLE IN PUBLIC REPO** | Evaluated via 2D occupancy & ray-cast projections (`unity_scene_map`) |

## Evaluation and Play Scripts Audit

- `navrep.scripts.cross_test_navreptrain_in_ianenv`: Evaluates policies trained on `NavRepTrainEnv` across all `IANEnv` scenarios (`irosasl1-6`, `irosasl_office_j1-6`, `irosunity_scene_map1-6`).
- `navrep.scripts.play_navreptrainenv`: Interactive viewer for `NavRepTrainEnv`.
- `navrep.scripts.play_ianenv`: Interactive viewer for `IANEnv`.
- `navrep.scripts.play_markenv`: Interactive viewer for `MarkEnv`.
- `navrep.scripts.play_navreptrainencodedenv`: Runs encoded policy in `NavRepTrainEncodedEnv`.
- `navrep.scripts.play_dreamenv`: Runs policy in world-model hallucinated "dream" environment.

