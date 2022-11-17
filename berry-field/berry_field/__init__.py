from gym.envs.registration import register

# the observation space is a Mx6 matrix in the form [isBerry, direction-vector, distance, size]
register(
    id='berry_field-v0',
    entry_point='berry_field.envs:BerryFieldEnv'
)

# imports for convenience 
from berry_field.envs import BerryFieldEnv
import berry_field.envs.analysis.visualization as visualization