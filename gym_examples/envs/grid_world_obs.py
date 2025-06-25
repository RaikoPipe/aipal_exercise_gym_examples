import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class GridWorldObsEnv(gym.Env):
    metadata = {'render.modes': ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, random_start=False, random_target=False,
                 reward_type="sparse", obstacles=None, num_obstacles=0, obstacle_density=0.0):
        self.size = size  # the size of the square grid
        self.window_size = 512  # size of the pygame window

        # Observations are dictionaries containing the agent's and target's positions
        # Each location is encoded as an element of {0, ..., "size"}^2, i.e. MultiDiscrete([size, size])
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "obstacles": spaces.Box(0, 1, shape=(size, size), dtype=int)
        })

        # We have 4 actions, corresponding to the 4 directions
        self.action_space = spaces.Discrete(4)

        # The following dictionary maps abstract actions from `self.action_space` to the direction we will walk
        # in if that action is taken.
        # I.e. 0 corresponds to "right", 1 to "up", 2 to "left", and 3 to "down".
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }

        assert render_mode is None or render_mode in self.metadata["render.modes"]
        self.render_mode = render_mode

        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.window = None
        self.clock = None

        # Adding class variables
        self.random_start = random_start
        self.random_target = random_target

        # Initializing the agent and target locations
        self._start_location = np.array([0, 0])
        self._agent_location = np.array([0, 0])
        self._target_location = np.array([0, 0])

        # Setting the reward type
        self.reward_type = reward_type

        # Obstacle configuration
        self.obstacles = obstacles  # List of [x, y] obstacle positions
        self.num_obstacles = num_obstacles  # Number of random obstacles
        self.obstacle_density = obstacle_density  # Density of random obstacles (0-1)

        # Initialize obstacle grid
        self._obstacle_grid = np.zeros((self.size, self.size), dtype=int)
        self._setup_obstacles()

    def _setup_obstacles(self):
        """Set up obstacles based on configuration."""
        self._obstacle_grid = np.zeros((self.size, self.size), dtype=int)

        # Add predefined obstacles
        if self.obstacles is not None:
            for obstacle in self.obstacles:
                x, y = obstacle
                if 0 <= x < self.size and 0 <= y < self.size:
                    self._obstacle_grid[x, y] = 1

        # Add random obstacles based on count
        if self.num_obstacles > 0:
            available_positions = []
            for x in range(self.size):
                for y in range(self.size):
                    if self._obstacle_grid[x, y] == 0:
                        available_positions.append([x, y])

            if len(available_positions) > 0:
                num_to_place = min(self.num_obstacles, len(available_positions))
                obstacle_positions = self.np_random.choice(
                    len(available_positions),
                    size=num_to_place,
                    replace=False
                )
                for idx in obstacle_positions:
                    x, y = available_positions[idx]
                    self._obstacle_grid[x, y] = 1

        # Add random obstacles based on density
        if self.obstacle_density > 0:
            for x in range(self.size):
                for y in range(self.size):
                    if self._obstacle_grid[x, y] == 0:  # Only place on empty cells
                        if self.np_random.random() < self.obstacle_density:
                            self._obstacle_grid[x, y] = 1

    def _is_valid_position(self, position):
        """Check if a position is valid (within bounds and not an obstacle)."""
        x, y = position
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        return self._obstacle_grid[x, y] == 0

    def _get_valid_random_position(self, avoid_positions=None):
        """Get a random valid position, avoiding specified positions."""
        if avoid_positions is None:
            avoid_positions = []

        valid_positions = []
        for x in range(self.size):
            for y in range(self.size):
                pos = np.array([x, y])
                if self._is_valid_position(pos):
                    # Check if position should be avoided
                    should_avoid = False
                    for avoid_pos in avoid_positions:
                        if np.array_equal(pos, avoid_pos):
                            should_avoid = True
                            break
                    if not should_avoid:
                        valid_positions.append(pos)

        if len(valid_positions) == 0:
            raise ValueError("No valid positions available")

        return valid_positions[self.np_random.integers(0, len(valid_positions))]

    def _get_obs(self):
        return {
            "agent": np.array(self._agent_location),
            "target": np.array(self._target_location),
            "obstacles": self._obstacle_grid.copy()
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location,
                ord=1
            ),
            "obstacles": self._obstacle_grid.copy()
        }

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Setup obstacles (may include random obstacles)
        self._setup_obstacles()

        if self.random_start:
            # Choose the agent's location uniformly at random from valid positions
            self._agent_location = self._get_valid_random_position()
        else:
            # Start the agent in the top left corner if valid, otherwise find valid position
            if self._is_valid_position(np.array([0, 0])):
                self._agent_location = np.array([0, 0])
            else:
                self._agent_location = self._get_valid_random_position()

        self._start_location = self._agent_location.copy()

        if self.random_target:
            # Sample the target's location randomly from valid positions, avoiding agent
            self._target_location = self._get_valid_random_position([self._agent_location])
        else:
            # Place the target in the bottom right corner if valid
            target_pos = np.array([self.size - 1, self.size - 1])
            if self._is_valid_position(target_pos) and not np.array_equal(target_pos, self._agent_location):
                self._target_location = target_pos
            else:
                self._target_location = self._get_valid_random_position([self._agent_location])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0, 1, 2, 3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # Calculate new position
        new_position = self._agent_location + direction

        # Only move if the new position is valid (within bounds and not an obstacle)
        if self._is_valid_position(new_position):
            self._agent_location = new_position

        # An episode is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        reward = self._compute_reward(direction, terminated)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _compute_reward(self, direction, terminated):
        if self.reward_type == "dense":
            # The reward is the negative Manhattan distance to the target
            reward = -(np.abs(self._target_location[0] - self._agent_location[0])
                       + np.abs(self._target_location[1] - self._agent_location[1]))
            # Add small penalty for hitting obstacles
            if not self._is_valid_position(self._agent_location + direction):
                reward -= 0.1
        else:
            # Binary sparse rewards with penalty for hitting obstacles
            if terminated:
                reward = 0
            elif not self._is_valid_position(self._agent_location + direction):
                reward = -1.1  # Slightly higher penalty for hitting obstacles
            else:
                reward = -1
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

        # First, we draw the obstacles
        for x in range(self.size):
            for y in range(self.size):
                if self._obstacle_grid[x, y] == 1:
                    pygame.draw.rect(
                        canvas,
                        (128, 128, 128),  # Gray color for obstacles
                        pygame.Rect(
                            pix_square_size * np.array([x, y]),
                            (pix_square_size, pix_square_size),
                            ),
                    )

        # Draw the start location
        pygame.draw.rect(
            canvas,
            (255, 255, 0),  # Yellow for start
            pygame.Rect(
                pix_square_size * self._start_location,
                (pix_square_size, pix_square_size),
                ),
        )

        # We draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 0),  # Green for target
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
                ),
        )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # Blue for agent
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


# Example usage:
if __name__ == "__main__":
    # Example 1: Predefined obstacles
    obstacles = [[1, 1], [2, 2], [3, 1]]
    env1 = GridWorldEnv(size=5, obstacles=obstacles, render_mode="human")

    # Example 2: Random obstacles by count
    env2 = GridWorldEnv(size=8, num_obstacles=10, render_mode="human")

    # Example 3: Random obstacles by density
    env3 = GridWorldEnv(size=10, obstacle_density=0.2, render_mode="human")

    # Example 4: Mixed configuration
    env4 = GridWorldEnv(
        size=7,
        obstacles=[[1, 1], [2, 2]],
        num_obstacles=3,
        obstacle_density=0.1,
        random_start=True,
        random_target=True,
        render_mode="human"
    )