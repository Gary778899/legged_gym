from pathlib import Path
import sys

try:
    from deploy.deploy_mujoco.deploy_mujoco_common import run
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from deploy.deploy_mujoco.deploy_mujoco_common import run


if __name__ == "__main__":
    run("pytorch")
