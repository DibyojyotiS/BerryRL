from copy import deepcopy

MAX_PARALLEL = 5
MAX_TRAIN_EPISODES = 500
GRID_SEARCH_CONFIG = {
    # add as <path.to.param>:[list of values]
    "seed":[2,4],
    "PER_BUFFER.alpha":[0.95, 0.7],
    "PER_BUFFER.beta":[0.5, 0.1],
    "TRAINING_STRAT_EPSILON_GREEDY.epsilon":[0.4, 0.2],
    "TRAINING_STRAT_EPSILON_GREEDY.finalepsilon":[0.1, 0.05],
}

BASE_CONFIG = {
    "seed": 4,
    "LOG_DIR_ROOT": ".temp/search/0.1",
    "run_name_prefix": "epsilons",
    "WANDB": dict(
        enabled = True, # set to true for server env
        project="agent-grid-search",
        group="grid-search-PER(alpha,beta)-epsilons",
        entity="foraging-rl",
        watch_log = "all", # logging both params and grads
        watch_log_freq = 1000,
        notes="""
        - The berries from memory and current observation are taken together 
            for the computation of the sectorized state. 
        - Random exploration action is enabled.
        - Reward Perception:
             scaled_clipped_reward = min(MAX, max(MIN, scale*actual_reward)) 
             actual_reward > 0: reward = 1 + nBerriesPicked/100 + scaled_clipped_reward
             actual_reward < 0: reward = scaled_clipped_reward
        - noise added all over the features INSIDE THE NN MODULE (added for every input parsed)
        - fixed the reward for exploration subroutine to the max-drain

        - Sectorized States:
            # a1: max-worth of each sector
            # a2: stores avg-worth of each sector
            # a3: a mesure of distance to max worthy in each sector
            # a4: indicates the sector with the max worthy berry
        """
        # TODO
        # Add -ve rewards based on the time-memories
    ),
    "RND_TRAIN_ENV": dict(
        field_size=(20000,20000), 
        patch_size=(2600,2600), 
        num_patches=10, 
        seperation=2400, 
        nberries=80, 
        initial_juice=0.5, 
        maxtime=5*60, 
        play_till_maxtime=False,
        patch_with_agent_at_center=True,
        end_on_boundary_hit= False,
        initial_pos_around_berry=True, 
        spawn_radius=100, 
    ),
    "AGENT": dict(
        skip_steps = 10,
        memory_config = dict(
            multiResTimeMemoryKwargs = dict(
                enabled = True,
                grid_sizes = [(20,20),(50,50),(100,100),(200,200),(400,400)],
                factor=0.6, 
                exp=1.0,
                persistence=0.8
            ),
            nearbyBerryMemoryKwargs = dict(
                enabled = True,
                minDistPopThXY=(1920/2, 1080/2), 
                maxDistPopThXY=(2600,2600), 
                memorySize=40
            ),
            localityMemoryKwargs = dict(
                enabled = False,
                resolution = (5,5)
            )
        ),
        state_computation_config = dict(
            persistence=0.8, 
            sector_angle=45,
            berryworth_offset=0.01,
            normalizing_berry_count = 200
        ),
        exploration_subroutine_config = dict(
            reward_discount_factor=0.99,
            max_steps=float('inf'),
            reward_type="discount-sum"
        ),
        reward_perception_config = dict(
            max_clip=float('inf'), min_clip=-float('inf'),
            scale=200,
            patch_dicovery_reward_config = dict(
                enabled = True,
                reward_value=2.0
            )
        ),
        nn_model_config = dict(
            layers=[32,16,16],
            lrelu_negative_slope=-0.001,
            noise=0.05
        )
    ),
    
    "ADAM": dict(
        lr=1e-4, weight_decay=0.0
    ),

    "MULTI_STEP_LR": dict(
        milestones=[100*i for i in range(1,3)],
        gamma=0.5
    ),

    "PER_BUFFER": dict(
        bufferSize=int(5E5), 
        alpha=0.95,
        beta=0.1, 
        beta_rate="$! (1 - valueOf('PER_BUFFER.beta'))/MAX_TRAIN_EPISODES}"
    ),

    "TRAINING_STRAT_EPSILON_GREEDY": dict(
        epsilon=0.55,
        finalepsilon=0.2,
        decaySteps=MAX_TRAIN_EPISODES,
        decay_type='exp'
    ),

    "DDQN": dict(
        batchSize=1024, 
        gamma=0.9, 
        update_freq=5, 
        MaxTrainEpisodes=MAX_TRAIN_EPISODES, 
        MaxStepsPerEpisode=None,
        optimize_every_kth_action=100,
        num_gradient_steps=25,
        evalFreq=10, 
        printFreq=1, 
        polyak_average=True, 
        polyak_tau=0.1,
        gradient_clip = 5,
        gradient_clipping_type = "norm",
    ),
}


def prepareConfig(base_config):

    def valueOf(path:str):
        nonlocal base_config
        root = base_config
        for k in path.split("."): 
            root = root[k]
        return root
    
    def dfsConfig(config:dict):
        nonlocal valueOf
        for key in config:
            val = config[key]
            if type(val) is dict:
                dfsConfig(config[key])
            elif type(val) is str and val.startswith("$!"):
                val = val[2:-1] # removes $!
                config[key] = eval(val)
    
    base_config_deepcopy = deepcopy(base_config)
    dfsConfig(base_config_deepcopy)
    return base_config_deepcopy

PARSED_BASE_CONFIG = prepareConfig(BASE_CONFIG)