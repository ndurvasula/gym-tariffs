from gym.envs.registration import register

register(
    id='tariffs-v0',
    entry_point='gym_tariffs.envs:TariffEnv',
)
