import time
import numpy as np

import mujoco
import mujoco.viewer

import gymnasium as gym
from gymnasium import spaces

from envs.interface import SimulatedRobot


class StackEnv(gym.Env):

    def __init__(self, xml_path='low_cost_robot/scene_two_cubes.xml', render=False):
        super(StackEnv, self).__init__()

        # Load the MuJoCo model and data
        self.model  = mujoco.MjModel.from_xml_path(xml_path)
        self.data   = mujoco.MjData(self.model)
        self.robot  = SimulatedRobot(self.model, self.data)

        self.do_render = render
        if self.do_render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.step_start = time.time()

        # Define the action space and observation space
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(3+1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.data.xpos.flatten().shape[0],), dtype=np.float32)

        # Initialize the robot and target positions
        self.target_pos = np.array([np.random.rand(), np.random.rand(), 0.1])
        self.threshold_distance = 0.05

    def reset(self):

        self.data.joint("red_box_joint").qpos[:3] = [np.random.rand()*0.2, np.random.rand()*0.2, 0.01]
        self.data.joint("blue_box_joint").qpos[:3] = [np.random.rand()*0.2, np.random.rand()*0.2, 0.01]

        mujoco.mj_step(self.model, self.data)

        self.step_start = time.time()
        return np.concatenate([self.data.xpos.flatten()])

    def reward(self):
        cube1_id = self.model.body("box1").id
        cube1_pos = self.data.geom_xpos[cube1_id]

        cube2_id = self.model.body("box2").id
        cube2_pos = self.data.geom_xpos[cube2_id]

        ### simplistic version of cube stacking reward
        return np.linalg.norm(cube1_pos[0:2] - cube2_pos[0:2]) < self.threshold_distance and cube1_pos[2] < cube2_pos[2]

    def step(self, action):

        # Update the robot position based on the action
        ee_id = self.model.body("joint5-pad").id
        ee_target_pos = self.data.xpos[ee_id] + action

        # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
        q_target_pos = self.robot.inverse_kinematics(ee_target_pos=ee_target_pos, joint_name="joint5-pad")
        q_target_pos[-1:] = np.sign(action[-1]) # Open or close the gripper
        self.robot.set_target_pos(q_target_pos)

        # Step the simulation forward
        mujoco.mj_step(self.model, self.data)

        # Compute the reward based on the distance
        reward = self.reward()
        done = reward

        # Return the next observation, reward, done flag, and additional info
        next_observation = np.concatenate([self.data.xpos.flatten()])
        
        return next_observation, float(reward), done, {}

    def render(self):
        if not self.do_render:
            return
        self.viewer.sync()
        time_until_next_step = self.model.opt.timestep - (time.time() - self.step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        self.step_start = time.time()
