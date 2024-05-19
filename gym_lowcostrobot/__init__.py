from gymnasium.envs.registration import register

__version__ = "0.0.1"

register(
    id="LiftCube-v0",
    entry_point="gym_lowcostrobot.envs:LiftCubeEnv",
    max_episode_steps=200,
)

register(
    id="PickPlaceCube-v0",
    entry_point="gym_lowcostrobot.envs:PickPlaceCubeEnv",
    max_episode_steps=200,
)

register(
    id="PushCube-v0",
    entry_point="gym_lowcostrobot.envs:PushCubeEnv",
    max_episode_steps=200,
)

register(
    id="ReachCube-v0",
    entry_point="gym_lowcostrobot.envs:ReachCubeEnv",
    max_episode_steps=200,
)

register(
    id="Stack-v0",
    entry_point="gym_lowcostrobot.envs:StackEnv",
    max_episode_steps=200,
)