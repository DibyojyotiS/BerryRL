CONFIG = {
    "seed": 4,
    "LOG_DIR_ROOT": ".temp/retrain/0.01",
    "run_name_prefix": "retrain-0.01",
    "WANDB": dict(
        enabled = True, # set to true for server env
        project="agent-design-v1",
        group=".temp/retrain/",
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
                memorySize=60
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
            reward_type="max-drain"
        ),
        reward_perception_config = dict(
            max_clip=float('inf'), min_clip=-float('inf'),
            scale=3 * 1e4/50
        ),
        nn_model_config = dict(
            layers=[32,16,16],
            lrelu_negative_slope=-0.001,
            noise=0.01
        )
    ),
    
    "ADAM": dict(
        lr=5e-4, weight_decay=0.0
    ),

    "MULTI_STEP_LR": dict(
        milestones=[100*i for i in range(1,3)],
        gamma=0.5
    ),

    "PER_BUFFER": dict(
        bufferSize=int(5E5), 
        alpha=0.95,
        beta=0.1, 
        beta_rate=0.9/2000
    ),

    "TRAINING_STRAT_EPSILON_GREEDY": dict(
        epsilon=0.55,
        finalepsilon=0.2,
        decaySteps=200,
        decay_type='exp'
    ),

    "DDQN": dict(
        batchSize=1024, 
        gamma=0.9, 
        update_freq=5, 
        MaxTrainEpisodes=2000, 
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