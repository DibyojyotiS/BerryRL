import gym

def make_berryField(observation_type = "ordered", bucket_angle = 45, reward_curiosity = False, 
                reward_curiosity_beta=0.25, reward_grid_size = (100,100), maxtime=5*60,
                agent_size=10):
    env = gym.make('berry_field:berry_field_mat_input-v0',
                   observation_type = observation_type,
                   reward_curiosity = reward_curiosity, 
                   reward_curiosity_beta=reward_curiosity_beta,
                   reward_grid_size = reward_grid_size, # should divide respective dimention of field_size
                   bucket_angle = bucket_angle,
                   maxtime = maxtime,
                   agent_size = agent_size
                   )
    return env
