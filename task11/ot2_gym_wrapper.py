"""

This module defines the OT2Env class, a custom Gymnasium environment for simulating
a robotic pipette using the PyBullet physics engine. The environment facilitates
interaction with the simulation by translating agent actions into simulation commands
and providing observations about the pipette's state and its goal position.
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_class import Simulation


class OT2Env(gym.Env):

    def __init__(self, seed, render=False, max_steps=1000):
        """
        Initializes the OT2Env environment.

        Parameters:
            render (bool): If True, enables rendering of the simulation GUI.
                           If False, runs the simulation in DIRECT mode without GUI.
            max_steps (int): The maximum number of steps allowed per episode.
        """
        super(OT2Env, self).__init__()
        self.enable_render = render
        self.max_steps = max_steps
        self.seed = seed

        # Initialize the Simulation with one agent and rendering as specified
        self.sim = Simulation(num_agents=1, render=self.enable_render)

        # Define the action space: 3 continuous actions (x, y, z velocities)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )

        # Define the observation space: 6 continuous observations (pipette x, y, z and goal x, y, z)
        self.observation_space = spaces.Box(
            low=-math.inf,
            high=math.inf,
            shape=(6,),
            dtype=np.float32
        )

        # Initialize step counter and robot identifier
        self.steps = 0
        self.robot_Id = None
        self.previous_distance = None

    @staticmethod
    def get_working_envelope_coords():
        """
        Retrieves the predefined working envelope coordinates.

        Returns:
            tuple: (x_min, x_max, y_min, y_max, z_min, z_max) defining the workspace boundaries.
        """
        working_envelope = {
            'corner_1': [-0.1881, -0.1713, 0.1195],
            'corner_2': [-0.1872, -0.1705, 0.2907],
            'corner_3': [-0.187, 0.2213, 0.1692],
            'corner_4': [-0.187, 0.2195, 0.2907],
            'corner_5': [0.2539, -0.171, 0.1694],
            'corner_6': [0.2532, -0.1705, 0.2906],
            'corner_7': [0.253, 0.2213, 0.1693],
            'corner_8': [0.253, 0.2195, 0.2907]
        }

        # Extract min and max values for each axis from the working envelope
        x_min = min(pos[0] for pos in working_envelope.values())
        x_max = max(pos[0] for pos in working_envelope.values())
        y_min = min(pos[1] for pos in working_envelope.values())
        y_max = max(pos[1] for pos in working_envelope.values())
        z_min = min(pos[2] for pos in working_envelope.values())
        z_max = max(pos[2] for pos in working_envelope.values())

        return x_min, x_max, y_min, y_max, z_min, z_max

    def reset(self, seed=None):
        """
        Resets the environment to an initial state and returns the initial observation.

        Parameters:
            seed (int, optional): Seed for the environment's random number generator to ensure reproducibility.

        Returns:
            tuple: (initial_observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)

        # Get workspace boundaries
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_working_envelope_coords()

        # Set a random goal position within the working envelope
        self.goal_position = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            np.random.uniform(z_min, z_max)
        ], dtype=np.float32)

        # Reset the simulation and obtain the initial observation
        observation = self.sim.reset(num_agents=1)

        # Extract the robot's unique identifier
        self.robot_Id = list(observation.keys())[0]

        # Retrieve the current pipette position
        pipette_pos = observation[self.robot_Id]['pipette_position']

        # Validate the pipette position length
        assert len(pipette_pos) == 3, f"Invalid pipette position: {pipette_pos}"

        # Combine pipette position with goal position to form the observation
        observation = np.array([
            *pipette_pos,
            *self.goal_position
        ], dtype=np.float32)

        # Reset step counter
        self.steps = 0
        self.previous_distance = np.linalg.norm(pipette_pos - self.goal_position)
        self.previous_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        return observation, {}

    def step(self, action):
        """
        Executes one time step within the environment.

        Parameters:
            action (np.ndarray): An action provided by the agent, consisting of [x_velocity, y_velocity, z_velocity].

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Append drop command (0 indicates no drop action)
        action = [*action, 0]

        # Run the simulation step with the given action
        observation = self.sim.run([action])

        # Ensure the robot identifier is defined
        assert self.robot_Id is not None, 'Robot Id is not defined'

        # Retrieve the current pipette position
        pipette_pos = observation[self.robot_Id]['pipette_position']

        # Validate the pipette position length
        assert len(pipette_pos) == 3, f"Invalid pipette position: {pipette_pos}"

        # Combine pipette position with goal position to form the observation
        observation = np.array([
            *pipette_pos,
            *self.goal_position
        ], dtype=np.float32)

        # # Calculate distance to goal
        # distance = np.linalg.norm(pipette_pos - self.goal_position)
        #
        # # Base reward is the negative distance
        # reward = -distance
        #
        # # Reward shaping based on progress
        # if self.previous_distance is not None:
        #     distance_reduction = self.previous_distance - distance
        #     progress_reward = 0.2 * distance_reduction  # Adjust the coefficient as needed
        #     reward += progress_reward
        #
        # self.previous_distance = distance  # Update for the next step
        #
        # # Bonus for reaching the goal
        # termination_threshold = 0.001  # Threshold in meters
        # if distance < termination_threshold:
        #     reward += 10  # Bonus for reaching the goal
        #     terminated = True
        # else:
        #     terminated = False
        #
        # # Time penalty to encourage faster solutions
        # time_penalty = -0.001  # Small negative reward each step
        # reward += time_penalty
        #
        # # Penalty for large actions to discourage erratic movements
        # action_magnitude = np.linalg.norm(action[:3])  # Considering only velocity components
        # action_penalty = -0.01 * action_magnitude  # Adjust the coefficient as needed
        # reward += action_penalty

        # Calculate distance to goal
        distance = np.linalg.norm(pipette_pos - self.goal_position)

        # Base reward: negative distance
        reward = -distance

        # Reward for reducing the distance
        if self.previous_distance is not None:
            distance_reduction = self.previous_distance - distance
            progress_reward = 0.2 * distance_reduction  # Increase weight for progress
            reward += progress_reward

        # Penalize overshooting for accuracy
        overshoot_penalty = 0.0
        for i in range(3):  # x, y, z axes
            if pipette_pos[i] > self.goal_position[i]:  # Overshot on this axis
                overshoot_penalty += (pipette_pos[i] - self.goal_position[i]) ** 2
        reward -= 0.5 * overshoot_penalty  # Adjust penalty scaling factor

        # Update previous distance
        self.previous_distance = distance

        # Bonus for reaching the goal
        termination_threshold = 0.001  # Accuracy requirement: 1 mm
        if distance < termination_threshold:
            reward += 20  # Higher bonus for reaching the goal accurately
            terminated = True
        else:
            terminated = False

        # Adjust time penalty to avoid rushing
        time_penalty = -0.0005  # Smaller penalty per step
        reward += time_penalty

        # Penalty for large or erratic actions
        action_magnitude = np.linalg.norm(action[:3])  # Considering velocity components
        action_penalty = -0.01 * action_magnitude
        reward += action_penalty

        # Determine if the episode should be truncated based on step count
        truncated = self.steps > self.max_steps

        # Increment step counter
        self.steps += 1

        # Return the observation, reward, termination flags, and additional info
        return observation, reward, terminated, truncated, {}

    def render(self, mode='human'):
        """
        Renders the environment. Since rendering is handled by PyBullet's GUI,
        this method can be extended for additional rendering functionalities if needed.

        Parameters:
            mode (str): The mode to render with. Currently, only 'human' is supported.

        Returns:
            None
        """
        pass  # Rendering is managed by the Simulation class

    def close(self):
        """
        Closes the environment and performs necessary cleanup.

        Returns:
            None
        """
        self.sim.close()
