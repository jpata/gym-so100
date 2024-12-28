import os

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import Env, spaces

from gym_so100 import ASSETS_PATH

BASE_LINK_NAME = "Rotation_Pitch"
EE_LINK_NAME = "Fixed_Jaw"


class PushCubeEnv(Env):
    """
    ## Description

    The robot has to push a cube with its end-effector.

    ## Action space

    Two action modes are available: "joint" and "ee". In the "joint" mode, the action space is a 6-dimensional box
    representing the target joint angles.

    | Index | Action              | Type (unit) | Min  | Max |
    | ----- | ------------------- | ----------- | ---- | --- |
    | 0     | Shoulder pan joint  | Float (rad) | -1.0 | 1.0 |
    | 1     | Shoulder lift joint | Float (rad) | -1.0 | 1.0 |
    | 2     | Elbow flex joint    | Float (rad) | -1.0 | 1.0 |
    | 3     | Wrist flex joint    | Float (rad) | -1.0 | 1.0 |
    | 4     | Wrist roll joint    | Float (rad) | -1.0 | 1.0 |
    | 5     | Gripper joint       | Float (rad) | -1.0 | 1.0 |

    In the "ee" mode, the action space is a 4-dimensional box representing the target end-effector position and the
    gripper position.

    | Index | Action        | Type (unit) | Min  | Max |
    | ----- | ------------- | ----------- | ---- | --- |
    | 0     | X             | Float (m)   | -1.0 | 1.0 |
    | 1     | Y             | Float (m)   | -1.0 | 1.0 |
    | 2     | Z             | Float (m)   | -1.0 | 1.0 |
    | 5     | Gripper joint | Float (rad) | -1.0 | 1.0 |

    ## Observation space

    The observation space is a dictionary containing the following subspaces:

    - `"agent_pos"`: the joint angles of the robot arm in radians, shape (6,)
    - `"agent_vel"`: the joint velocities of the robot arm in radians per second, shape (6,)
    - `"target_pos"`: the position of the target, as (x, y, z)
    - `"image_front"`: the front image of the camera of size (240, 320, 3)
    - `"image_top"`: the top image of the camera of size (240, 320, 3)
    - `"environment_state"`: the position of the cube, as (x, y, z)

    Three observation modes are available: "image" (default), "state", and "both".

    | Key             | `"image"` | `"state"` | `"both"` |
    | --------------- | --------- | --------- | -------- |
    | `"agent_pos"`    | ✓         | ✓         | ✓        |
    | `"agent_vel"`    | ✓         | ✓         | ✓        |
    | `"target_pos"`  | ✓         | ✓         | ✓        |
    | `"image_front"` | ✓         |           | ✓        |
    | `"image_top"`   | ✓         |           | ✓        |
    | `"environment_state"`    |           | ✓         | ✓        |

    ## Reward

    The reward is the negative distance between the cube and the target position.

    ## Arguments

    - `observation_mode (str)`: the observation mode, can be "image", "state", or "both", default is "image", see
        section "Observation space".
    - `action_mode (str)`: the action mode, can be "joint" or "ee", default is "joint", see section "Action space".
    - `render_mode (str)`: the render mode, can be "human" or "rgb_array", default is None.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self, observation_mode="image", action_mode="joint", render_mode=None):
        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(os.path.join(ASSETS_PATH, "push_cube.xml"), {})
        self.data = mujoco.MjData(self.model)

        # Set the action space
        self.action_mode = action_mode
        action_shape = {"joint": 6, "ee": 4}[action_mode]
        self.action_space = spaces.Box(low=-3, high=3, shape=(action_shape,), dtype=np.float32)
 
        self.nb_dof = 6

        # Set the observations space
        self.observation_mode = observation_mode
        observation_subspaces = {
            "agent_pos": spaces.Box(low=-np.pi, high=np.pi, shape=(6,)),
            "agent_vel": spaces.Box(low=-10.0, high=10.0, shape=(6,)),
            "target_pos": spaces.Box(low=-10.0, high=10.0, shape=(3,)),
        }
        if self.observation_mode in ["image", "both"]:
            observation_subspaces["image_front"] = spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8)
            observation_subspaces["image_top"] = spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8)
            self.renderer = mujoco.Renderer(self.model)
        if self.observation_mode in ["state", "both"]:
            observation_subspaces["environment_state"] = spaces.Box(low=-10.0, high=10.0, shape=(3,))

        self.observation_space = gym.spaces.Dict(observation_subspaces)

        # Set the render utilities
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.azimuth = -75
            self.viewer.cam.distance = 1
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer = mujoco.Renderer(self.model, height=640, width=640)

        # Set additional utils
        self.threshold_height = 0.5
        self.cube_low = np.array([-0.15, -0.45, 0.015])
        self.cube_high = np.array([0.15, -0.35, 0.015])
        self.target_low = np.array([-0.15, -0.45, 0.005])
        self.target_high = np.array([0.15, -0.35, 0.2])

        # get dof addresses
        self.cube_dof_id = self.model.body("cube").dofadr[0]
        self.arm_dof_id = self.model.body(BASE_LINK_NAME).dofadr[0]
        self.arm_dof_vel_id = self.arm_dof_id
        # if the arm is not at address 0 then the cube will have 7 states in qpos and 6 in qvel
        if self.arm_dof_id != 0:
            self.arm_dof_id = self.arm_dof_vel_id + 1

        self.control_decimation = 4 # number of simulation steps per control step

    def apply_action(self, action):
        """
        Step the simulation forward based on the action

        Action shape
        - EE mode: [dx, dy, dz, gripper]
        - Joint mode: [q1, q2, q3, q4, q5, q6, gripper]
        """
        target_low = np.array([-2.2, -3.14158, 0, -2.0, -3.14158, -0.2])
        target_high = np.array([2.2, 0.2, 3.14158, 1.8, 3.14158, 2.0])
        target_qpos = np.deg2rad(np.array(action))#.clip(target_low, target_high)

        # Set the target position
        self.data.ctrl = target_qpos

        # Step the simulation forward
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
            if self.render_mode == "human":
                self.viewer.sync()


    def get_observation(self):
        # pos is [x, y, z, qw, qx, qy, qz, q1, q2, q3, q4, q5, q6, gripper]
        # vel is [vx, vy, vz, wx, wy, wz, dq1, dq2, dq3, dq4, dq5, dq6, dgripper]
        observation = {
            "agent_pos": self.data.qpos[self.arm_dof_id:self.arm_dof_id+self.nb_dof].astype(np.float32),
            "agent_vel": self.data.qvel[self.arm_dof_vel_id:self.arm_dof_vel_id+self.nb_dof].astype(np.float32),
            "target_pos": self.target_pos,
        }
        if self.observation_mode in ["image", "both"]:
            self.renderer.update_scene(self.data, camera="camera_front")
            observation["image_front"] = self.renderer.render()
            self.renderer.update_scene(self.data, camera="camera_top")
            observation["image_top"] = self.renderer.render()
        if self.observation_mode in ["state", "both"]:
            observation["environment_state"] = self.data.qpos[self.cube_dof_id:self.cube_dof_id+3].astype(np.float32)
        return observation

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Reset the robot to the initial position and sample the cube position
        cube_pos = self.np_random.uniform(self.cube_low, self.cube_high)
        cube_rot = np.array([1.0, 0.0, 0.0, 0.0])
        robot_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.arm_dof_id:self.arm_dof_id+self.nb_dof] = robot_qpos
        self.data.qpos[self.cube_dof_id:self.cube_dof_id+7] = np.concatenate([cube_pos, cube_rot])

        # Sample the target position
        self.target_pos = self.np_random.uniform(self.target_low, self.target_high).astype(np.float32)

        # update visualization
        self.model.geom('target_region').pos = self.target_pos[:]

        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        return self.get_observation(), {}

    def step(self, action):
        # Perform the action and step the simulation
        self.apply_action(action)

        # Get the new observation
        observation = self.get_observation()

        # Get the position of the cube and the distance between the end effector and the cube
        cube_pos = self.data.qpos[self.cube_dof_id:self.cube_dof_id+3]
        ee_pos = self.data.qpos[self.arm_dof_id:self.arm_dof_id+3]
        cube_to_target = np.linalg.norm(cube_pos - self.target_pos)
        cube_to_ee = np.linalg.norm(cube_pos - ee_pos)
        ee_to_target = np.linalg.norm(ee_pos - self.target_pos)

        # Compute the reward
        # reward = -(cube_to_target + cube_to_ee)
        reward = -(ee_to_target)
        return observation, reward, False, False, {}

    def render(self):
        if self.render_mode == "human":
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer.update_scene(self.data, camera="camera_vizu")
            return self.rgb_array_renderer.render()

    def close(self):
        if self.render_mode == "human":
            self.viewer.close()
        if self.observation_mode in ["image", "both"]:
            self.renderer.close()
        if self.render_mode == "rgb_array":
            self.rgb_array_renderer.close()
