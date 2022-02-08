from gym.envs.registration import register

# the observation space is a Mx6 matrix in the form [isBerry, direction-vector, distance, size]
register(
    id='berry_field_mat_input-v0',
    entry_point='berry_field.envs:BerryFieldEnv_MatInput'
)