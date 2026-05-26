# Project Instructions for Codex (Repository-level)

## Project context
- Repository: humanoid robot / legged locomotion
- Languages: Python, C++
- Frameworks: ROS2 Humble, CMake
- Code types: simulation, deployment, training, real robot scripts
- Development environment: localhost Ubuntu 24.04 without ROS2
- Mock test environment: Docker container Ubuntu 22.04 with ROS Humble, for ONNX model and MuJoCo through middleware
- Server: 4x3090 GPU, accessed via SSH
- Personal workstation: ultra9-285k + RTX 5909D + 48G RAM

## Python environment (local)
- Local dev venv: `${workspaceFolder}/.venv`
- Activate before running scripts or tests: `source .venv/bin/activate`
- Run tests with: `.venv/bin/python -m pytest`
- Do not rely on global Python environment

## Python environment (container)
- Container Python is already set as global venv.
- ROS2 environment has been automatically sourced via Dockerfile.
- No manual activation is required before running scripts or tests.
- Run tests with: `python -m pytest` (no need to prepend .venv)

## Docs generation rules
- Create .md file as default unless specified
- If the doc is classified into top-level, e.g. related to project structure, guidance or deployment, put the generated doc in docs/
- If the doc is classfied into test records or notes, name the doc in the format as "RECORD_2026-05-20_MIDDLEWARE_ACCEPTANCE"
- If the doc is classfied into summaries, name the doc in the format as "SUMMARY_2026-05-20_MIDDLEWARE_PHASE7_HARDENING"
- If the doc is classfied into design and ideas, put the generated doc in docs/design
- If you are not sure about where to place the docs or how to name the docs, please ask me first