# Simulation Startup Summary Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the independent DexSim implementation and focused review. Execute the EmbodiChain integration in this session.

**Goal:** Give every EmbodiChain simulation a readable, highlighted rendering/physics/runtime summary and a DexSim startup-info switch.

**Architecture:** DexSim owns `WorldConfig.log_startup_info: bool = True` and gates only informational startup messages at their sources. EmbodiChain owns compact/full/off summary configuration, source-derived rows, terminal-aware table formatting, standalone lifecycle emission and Gym summary integration.

**Tech Stack:** Python, existing PrettyTable/logger, C++17, pybind11, pytest, CMake.

**Spec:** Approved conversation design, with Engine threads and Stepping removed. Default solver must name the verified native algorithm, not claim automatic selection.

## Global Constraints

- Work in current feature branches; preserve unrelated changes. No commits or publication in this task.
- Keep warnings/errors visible. Do not globally redirect or mute stdout/stderr.
- Do not claim rendering is disabled merely because the native window is closed.
- Do not claim CUDA Graph capture before runtime capture.
- Unit tests should not initialize GPU hardware. Run live tutorials serially.

## Task 1: DexSim startup logging

- [x] Add failing subprocess tests for default-on/explicit-off startup output and retained warning/error behavior.
- [x] Add WorldConfig/EngineConfig flag, conversion, binding/docstrings and Python Newton access; gate EngineConfig dump, CUDA/Vulkan info and Newton informational startup selection.
- [x] Verify Default solver implementation and provide the exact algorithm/source to EmbodiChain integration.
- [x] Build cached release targets with the cached Python interpreter, run tests, update routed context.

## Task 2: EmbodiChain summary

Files: sim/sim_manager.py, sim/_startup_summary.py, physics/{base,default,newton}.py, utils/logger.py, gym/envs/base_env.py and focused tests.

- [x] Add failing tests for compact/full/off, no Engine threads/Stepping, backend-specific values, pending solver/graph, color and no-color, once-only emission.
- [x] Add `startup_summary="compact"` and `dexsim_startup_info=False` to SimulationManagerCfg and forward the latter before World construction.
- [x] Build a small private summary module; use existing PrettyTable and one plain logger record. Read configured vs runtime values explicitly.
- [x] Print standalone core summary after construction and scene summary at first use; defer Gym output to its existing complete initialization boundary and combine common rows with task/manager information.
- [x] Run focused simulation, config, logger and Gym tests; update relevant context/API documentation.

## Task 3: Joint validation and review

- [x] Re-run create_scene Default CPU and GPU plus Newton articulation/robot headlessly against the rebuilt DexSim.
- [x] Verify bottom-layer info toggles, warning visibility, no duplicate main tables, non-TTY output, and actual highlighted terminal rendering.
- [x] Review both diffs, resolve actionable findings, report results and any limitations.

## Verification record

- DexSim release Python package built; 73 related tests passed before the final context-success gate; all 9 updated startup subprocess checks also passed against the final build.
- EmbodiChain focused manager/summary/logger/Gym timing: 131 passed; config parsing: 94 passed.
- Four headless tutorial configurations passed; live Gym full/off and terminal/NO_COLOR behavior checked.
- Baseline camera-factory and missing-physics-config test failures reproduced from an independent HEAD export; unrelated behavior left unchanged.
- Review fixes: avoid diagnostic PyTorch CUDA initialization; only successful preparation can publish READY.
- No commits or publication performed.
