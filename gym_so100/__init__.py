import os

from gymnasium.envs.registration import register

__version__ = "0.0.1"

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "assets", "trs_so_arm100")
BASE_LINK_NAME = "Base"
register(
    id="gym_so100/PushCube-v0",
    entry_point="gym_so100.envs:PushCubeEnv",
    max_episode_steps=500,
)
