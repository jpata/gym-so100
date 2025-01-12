import os

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import Env, spaces

from gym_so100 import ASSETS_PATH

BASE_LINK_NAME = "Rotation_Pitch"
EE_LINK_NAME = "Fixed_Jaw"


def get_collision_info(model, data):
    """
    Get collision information from the MuJoCo simulation.
    
    Args:
        model (mujoco.MjModel): The MuJoCo model
        data (mujoco.MjData): The MuJoCo simulation data
    
    Returns:
        dict: Dictionary containing collision information
            - contacts: List of active contacts
            - contact_forces: Contact forces for each collision
            - colliding_bodies: Pairs of bodies that are in contact
    """
    collision_info = {
        'contacts': [],
        'contact_forces': [],
        'colliding_bodies': set()
    }
    
    # Iterate through all contacts in the simulation
    for i in range(data.ncon):
        contact = data.contact[i]
        
        # Get the geom names involved in the contact
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        geom1_name = model.geom(geom1_id).name
        geom2_name = model.geom(geom2_id).name
        
        # Get the contact force magnitude
        force_magnitude = np.linalg.norm(data.efc_force[data.contact[i].efc_address:data.contact[i].efc_address + contact.dim])
        
        contact_info = {
            'geom1': geom1_name,
            'geom2': geom2_name,
            'pos': contact.pos.copy(),
            'force': force_magnitude,
            'friction': contact.friction.copy()
        }
        
        collision_info['contacts'].append(contact_info)
        collision_info['contact_forces'].append(force_magnitude)
        collision_info['colliding_bodies'].add((geom1_name, geom2_name))
    
    return collision_info

def get_joint_forces(data):
    """
    Get the total force on all joints in the MuJoCo simulation.
    
    Args:
        data (mujoco.MjData): The MuJoCo simulation data object
    
    Returns:
        dict: Dictionary containing different force measurements for each joint
            - qfrc_actuator: Actuator forces
            - qfrc_bias: Bias forces (Coriolis, centrifugal, gravitational)
            - qfrc_constraint: Constraint forces
            - qfrc_applied: Externally applied forces
            - qfrc_passive: Spring, damping, friction forces
    """
    joint_names = [
        "Rotation",
        "Pitch",
        "Elbow",
        "Wrist_Pitch",
        "Wrist_Roll",
        "Jaw"
    ]
    
    # Get the indices for each joint
    joint_indices = []
    for name in joint_names:
        joint_id = data.joint(name).id
        joint_indices.append(joint_id)
    
    forces = {
        'qfrc_actuator': {},    # Actuator forces
        'qfrc_bias': {},        # Bias forces (Coriolis, centrifugal, gravitational)
        'qfrc_constraint': {},  # Constraint forces
        'qfrc_applied': {},     # Externally applied forces
        'qfrc_passive': {},     # Spring, damping, friction forces
    }
    
    # Collect forces for each joint
    for name, idx in zip(joint_names, joint_indices):
        forces['qfrc_actuator'][name] = data.qfrc_actuator[idx]
        forces['qfrc_bias'][name] = data.qfrc_bias[idx]
        forces['qfrc_constraint'][name] = data.qfrc_constraint[idx]
        forces['qfrc_applied'][name] = data.qfrc_applied[idx]
        forces['qfrc_passive'][name] = data.qfrc_passive[idx]
    
    # Calculate total force for each joint
    total_forces = {}
    for name in joint_names:
        total_forces[name] = (
            forces['qfrc_actuator'][name] +
            forces['qfrc_bias'][name]
            # forces['qfrc_constraint'][name] +
            # forces['qfrc_applied'][name] +
            # forces['qfrc_passive'][name]
        )
    
    forces['total'] = total_forces
    
    return forces

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
    - `"environment_state"`: the position of the cube and target as (x, y, z)

    Three observation modes are available: "image" (default), "state", and "both".

    | Key             | `"image"` | `"state"` | `"both"` |
    | --------------- | --------- | --------- | -------- |
    | `"agent_pos"`   | ✓         | ✓         | ✓        |
    | `"agent_vel"`   | ✓         | ✓         | ✓        |
    | `"target_pos"`  | ✓         | ✓         | ✓        |
    | `"image_front"` | ✓         |           | ✓        |
    | `"image_top"`   | ✓         |           | ✓        |
    | `"env_state"`   |           | ✓         | ✓        |

    ## Reward

    The reward is the negative distance between the cube and the target position.

    ## Arguments

    - `observation_mode (str)`: the observation mode, can be "image", "state", or "both", default is "image", see
        section "Observation space".
    - `action_mode (str)`: the action mode, can be "joint" or "ee", default is "joint", see section "Action space".
    - `render_mode (str)`: the render mode, can be "human" or "rgb_array", default is None.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, observation_mode="image", action_mode="joint", render_mode=None, curriculum_level=0):
        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(os.path.join(ASSETS_PATH, "push_cube.xml"), {})
        self.data = mujoco.MjData(self.model)

        # Set the action space
        self.action_mode = action_mode
        self.action_shape = 6
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_shape,), dtype=np.float32)
        self.nb_dof = 6

        self.joint_limits = {
            'low': np.array([-2.2, -3.14158, 0, -2.0, -3.14158, -0.2]),
            'high': np.array([2.2, 0.2, 3.14158, 1.8, 3.14158, 2.0])
        }

        # Curriculum settings
        self.curriculum_level = curriculum_level        

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
            observation_subspaces["environment_state"] = spaces.Box(low=-10.0, high=10.0, shape=(6,))

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
        self.cube_low = np.array([-0.1, -0.1, 0.005])
        self.cube_high = np.array([0.1, -0.1, 0.005])
        self.target_low = np.array([-0.15, -0.25, 0.005])
        self.target_high = np.array([0.15, -0.15, 0.005])

        # get dof addresses
        self.cube_dof_id = self.model.body("cube").dofadr[0]
        self.arm_dof_id = self.model.body(BASE_LINK_NAME).dofadr[0]
        self.arm_dof_vel_id = self.arm_dof_id

        self.control_decimation = 20 # number of simulation steps per control step

    def apply_action(self, action):
        """
        Step the simulation forward based on the action.
        Actions must be radians, in the allowed range of the joints.
        Note that when training, the policy must normalize the outputs to the allowed range
        """

        current_joint_angles = self.data.qpos[self.arm_dof_id:self.arm_dof_id+self.nb_dof]
        self.attempted_joint_pos = action + current_joint_angles
        target_arm_qpos = np.clip(
            self.attempted_joint_pos,
            self.joint_limits['low'],
            self.joint_limits['high'],
        )

        self.data.ctrl = target_arm_qpos

        # Step the simulation forward
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
            if self.render_mode == "human":
                self.viewer.sync()


    def get_observation(self):
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
            observation["environment_state"] = np.concatenate([
                self.data.qpos[self.cube_dof_id:self.cube_dof_id+3].astype(np.float32),
                self.target_pos.astype(np.float32)
            ])
        return observation

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)
        
        params = self.get_curriculum_parameters()
        
        # Sample positions based on current curriculum ranges
        cube_range = params["cube_pos_range"]
        target_range = params["target_pos_range"]

        # Keep sampling positions until minimum distance constraint is satisfied
        while True:
            # Sample target position first
            self.target_pos = self.np_random.uniform(
                target_range["low"],
                target_range["high"]
            ).astype(np.float32)
            
            # Sample cube position
            cube_pos = self.target_pos + self.np_random.uniform(
                cube_range["low"],
                cube_range["high"]
            ).astype(np.float32)
            
            # Check if distance constraint is satisfied
            distance = np.linalg.norm(cube_pos[:2] - self.target_pos[:2])  # Only check x-y distance
            if distance >= 0.015:
                break


        cube_rot = np.array([1.0, 0.0, 0.0, 0.0])
        robot_qpos = np.array([0.0, -3.14, 3, 1.24, 0.0, 0.0])
        self.data.qpos[self.arm_dof_id:self.arm_dof_id+self.nb_dof] = robot_qpos
        self.data.qpos[self.cube_dof_id:self.cube_dof_id+7] = np.concatenate([cube_pos, cube_rot])

        # Sample the target position

        # update visualization
        self.model.geom('target_region').pos = self.target_pos[:]

        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        return self.get_observation(), {}

    def get_curriculum_parameters(self):
        """
        Get current curriculum parameters based on level.
        Each level increases task complexity across multiple dimensions.
        """
        params = {
            # Base workspace boundaries
            "cube_pos_range": None,
            "target_pos_range": None,
            "required_height": None,
            "placement_threshold": None,
            "time_limit": None,
            "reward_scales": None
        }
        
        # Level 0: Simple reaching task
        if self.curriculum_level == 0:
            params.update({
                "cube_pos_range": {
                    "low": np.array([-0.03, -0.03, 0.0]),
                    "high": np.array([0.03, 0.03, 0.0])
                },
                "target_pos_range": {
                    "low": np.array([-0.05, -0.22, 0.01]),
                    "high": np.array([0.05, -0.18, 0.01])
                },
                "required_height": 0.0,  # No lifting required
                "placement_threshold": 0.08,  # Larger tolerance
                "time_limit": 100,
                "reward_scales": {
                    "reaching": 1.0,
                    "grasping": 0.5,
                    "lifting": 0.0,  # No lifting reward yet
                    "target": 0.3,
                    "success": 2.0
                }
            })
            
        # Level 1: Basic grasping
        elif self.curriculum_level == 1:
            params.update({
                "cube_pos_range": {
                    "low": np.array([-0.05, -0.05, 0.0]),
                    "high": np.array([0.05, 0.05, 0.0])
                },
                "target_pos_range": {
                    "low": np.array([-0.10, -0.20, 0.01]),
                    "high": np.array([0.10, -0.13, 0.01])
                },
                "required_height": 0.02,  # Slight lifting
                "placement_threshold": 0.06,
                "time_limit": 150,
                "reward_scales": {
                    "reaching": 0.8,
                    "grasping": 1.0,
                    "lifting": 0.3,
                    "target": 0.4,
                    "success": 3.0
                }
            })
            
        # Level 2: Basic lifting
        elif self.curriculum_level == 2:
            params.update({
                "cube_pos_range": {
                    "low": np.array([-0.07, -0.07, 0.0]),
                    "high": np.array([0.07, 0.07, 0.0])
                },
                "target_pos_range": {
                    "low": np.array([-0.12, -0.22, 0.01]),
                    "high": np.array([0.12, -0.11, 0.01])
                },
                "required_height": 0.05,  # Significant lifting
                "placement_threshold": 0.05,
                "time_limit": 200,
                "reward_scales": {
                    "reaching": 0.6,
                    "grasping": 1.0,
                    "lifting": 0.8,
                    "target": 0.6,
                    "success": 4.0
                }
            })
            
        # Level 3: Extended workspace
        elif self.curriculum_level == 3:
            params.update({
                "cube_pos_range": {
                    "low": np.array([-0.09, -0.09, 0.0]),
                    "high": np.array([0.09, 0.09, 0.0])
                },
                "target_pos_range": {
                    "low": np.array([-0.15, -0.25, 0.01]),
                    "high": np.array([0.15, -0.1, 0.01])
                },
                "required_height": 0.08,
                "placement_threshold": 0.04,
                "time_limit": 250,
                "reward_scales": {
                    "reaching": 0.4,
                    "grasping": 1.0,
                    "lifting": 1.0,
                    "target": 0.8,
                    "success": 5.0
                }
            })
            
        # Level 4: Precise placement
        elif self.curriculum_level == 4:
            params.update({
                "cube_pos_range": {
                    "low": np.array([-0.14, -0.14, 0.0]),
                    "high": np.array([0.14, 0.14, 0.0])
                },
                "target_pos_range": {
                    "low": np.array([-0.2, -0.3, 0.01]),
                    "high": np.array([0.2, -0.1, 0.01])
                },
                "required_height": 0.1,
                "placement_threshold": 0.03,
                "time_limit": 300,
                "reward_scales": {
                    "reaching": 0.3,
                    "grasping": 1.0,
                    "lifting": 1.0,
                    "target": 1.0,
                    "success": 6.0
                }
            })
            
        # Level 5: Full task
        else:
            params.update({
                "cube_pos_range": {
                    "low": np.array([-0.16, -0.16, 0.0]),
                    "high": np.array([0.16, 0.16, 0.0])
                },
                "target_pos_range": {
                    "low": np.array([-0.25, -0.35, 0.01]),
                    "high": np.array([0.25, -0.1, 0.01])
                },
                "required_height": 0.12,
                "placement_threshold": 0.02,
                "time_limit": 400,
                "reward_scales": {
                    "reaching": 0.2,
                    "grasping": 1.0,
                    "lifting": 1.0,
                    "target": 1.2,
                    "success": 8.0
                }
            })
            
        return params

    def compute_reward(self, observation, action):
        """
        Compute reward with curriculum-adjusted scales and requirements
        """
        params = self.get_curriculum_parameters()
        reward_scales = params["reward_scales"]
        required_height = params["required_height"]
        placement_threshold = params["placement_threshold"]
        
        # Get positions and collisions
        cube_pos = self.data.qpos[self.cube_dof_id:self.cube_dof_id+3]
        ee_id = self.model.site("end_effector_site").id
        ee_pos = self.data.qpos[ee_id:ee_id+3]
        
        collision_info = get_collision_info(self.model, self.data)
        cb = collision_info["colliding_bodies"]
        
        reward = 0.0
        
        # Reaching reward
        ee_to_cube = np.linalg.norm(cube_pos - ee_pos)
        reward += reward_scales["reaching"] * -ee_to_cube
        
        # Grasping reward
        grasping_cube = any((pad, "red_box") in cb for pad in [
            "fixed_jaw_pad_1", "fixed_jaw_pad_2", "fixed_jaw_pad_3", "fixed_jaw_pad_4",
            "moving_jaw_pad_1", "moving_jaw_pad_2", "moving_jaw_pad_3", "moving_jaw_pad_4"
        ])
        if grasping_cube:
            reward += reward_scales["grasping"]
        
        # Lifting reward
        cube_height = cube_pos[2]
        if required_height > 0 and cube_height > required_height:
            lift_reward = (cube_height - required_height) / required_height
            reward += reward_scales["lifting"] * lift_reward
        
        # Target reward
        cube_to_target = np.linalg.norm(cube_pos - self.target_pos)
        height_factor = max(0.1, min(1.0, cube_height / required_height if required_height > 0 else 0.0))
        target_reward = -cube_to_target * height_factor
        reward += reward_scales["target"] * target_reward
        
        # Success reward
        if (cube_to_target < placement_threshold and 
            abs(cube_height - self.target_pos[2]) < placement_threshold):
            reward += reward_scales["success"]
        
        # Penalties (constant across curriculum)
        collide_floor = any(("floor", pad) in cb for pad in [
            "fixed_jaw_pad_1", "fixed_jaw_pad_2", "fixed_jaw_pad_3", "fixed_jaw_pad_4",
            "moving_jaw_pad_1", "moving_jaw_pad_2", "moving_jaw_pad_3", "moving_jaw_pad_4"
        ])
        if collide_floor:
            reward -= 1.0
        
        forces = get_joint_forces(self.data)
        mag_total_force = sum(abs(v) for v in forces["total"].values())
        reward -= 0.001 * mag_total_force
        
        reward -= 0.001 * np.sum(np.abs(action))

        # Add joint limit penalty based on attempted position
        # Calculate how much the attempted position would have exceeded limits
        lower_violation = np.maximum(0, self.joint_limits['low'] - self.attempted_joint_pos)
        upper_violation = np.maximum(0, self.attempted_joint_pos - self.joint_limits['high'])
        total_violation = lower_violation + upper_violation
        
        # Apply quadratic penalty for attempted joint limit violations
        joint_limit_penalty = 2.0 * np.sum(total_violation ** 2)
        reward -= joint_limit_penalty

        return reward

    def step(self, action):
        self.apply_action(action)
        observation = self.get_observation()
        reward = self.compute_reward(observation, action)

        cube_pos = self.data.qpos[self.cube_dof_id:self.cube_dof_id+3]
        cube_to_target = np.linalg.norm(cube_pos - self.target_pos)
        success = (cube_to_target < 0.01 and 
            abs(cube_pos[2] - self.target_pos[2]) < 0.01)
        terminated = success
        truncated = False
        if success:
            print("success")
        info = {"is_success": success}

        return observation, reward, terminated, truncated, info


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
