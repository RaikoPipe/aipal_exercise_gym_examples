import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, random_start=False, random_target=False, reward_type="sparse"):
        self.size = size  # the size of the square grid
        self.window_size = 512  # size of the pygame window

        # Observations are dictionaries containing the agent's positions
        # Each location is encoded as an element of {0, ..., "size"}^2, i.e. MultiDiscrete([size, size])

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=int)
        }
        )

        # We gave 4 action, corresponsing to the 4 directions
        self.action_space = spaces.Discrete(4)

        """The following dictionary maps abstract actions from `self.action_space` to the direction we will walk
        in if that action is taken. 
        I.e. 0 corresponds to "right", 1 to "up", 2 to "left", and 3 to "down".
        """

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }

        assert render_mode is None or render_mode in self.metadata["render.modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """

        self.window = None
        self.clock = None

        # adding class variables
        self.random_start = random_start
        self.random_target = random_target

        # initialising the agent and target locations
        self._start_location = [0,0]
        self._agent_location = [0,0]
        self._target_location = [0,0]

        # setting the reward type
        self.reward_type = reward_type

    def _get_obs(self):
        return {"agent": np.array(self._agent_location), "target": np.array(self._target_location)}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location,
                ord=1
            )
        }

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if self.random_start:
            # Choose the agent's location uniformly at random
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else:
            # Start the agent in the top left corner
            self._agent_location = np.array([0, 0])

        self._start_location = self._agent_location

        if self.random_target:
            # We will sample the target's location randomly until it does not coincide with the agent's location
            self._target_location = self._agent_location
            while np.array_equal(self._target_location, self._agent_location):
                self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else:
            # Place the target in the bottom right corner
            self._target_location = np.array([self.size - 1, self.size - 1])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0, 1, 2, 3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't walk off the grid world
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # An episode is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        reward = self._compute_reward(terminated)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _compute_reward(self, terminated):
        if self.reward_type == "dense":
            # The reward is the negative Manhattan distance to the target
            reward = -(np.abs(self._target_location[0] - self._agent_location[0])
                       + np.abs(self._target_location[1] - self._agent_location[1]))

        else:
            reward = 0 if terminated else -1  # binary sparse rewards
        return reward

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First, we draw the start location
        pygame.draw.rect(
            canvas,
            (255, 255, 0),
            pygame.Rect(
                pix_square_size * self._start_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # We draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
