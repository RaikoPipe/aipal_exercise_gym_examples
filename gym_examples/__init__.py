from gymnasium.envs.registration import register

register(
     id="gym_examples/GridWorld-v0",
     entry_point="gym_examples.envs:GridWorldEnv",
     max_episode_steps=100,
)

register(
     id="gym_examples/GridWorldObs-v0",
     entry_point="gym_examples.envs:GridWorldObsEnv",
        max_episode_steps=100,
)