CONFIG = {
    "seed": 4,
    "LOG_DIR_ROOT": ".temp/nearby-berry-memory/0.0.1",
    "WANDB": dict(
        enabled = True, # set to true for server env
        project="agent-design-v1",
        group="nearby-berry-memory/0.0.1",
        entity="foraging-rl",
        watch_log_freq = 100,
        notes=(
            "the berries from memory and current observation are taken" 
            + "together for the computation of the sectorized state,"
            + "the maxObsDiag parameter is adjusted to the maximum limit of memory"
            + "this allows enough wiggle of values in the sectorized state."
            + "random exploration action is enabled"
            )
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
                grid_sizes = [(20,20),(50,50),(100,100),(200,200),(400,400)],
                factor=0.6, 
                exp=1.0,
            ),
            nearbyBerryMemoryKwargs = dict(
                minDistPopThXY=(1920/2, 1080/2), 
                maxDistPopThXY=(2600,2600), 
                memorySize=50
            )
        ),
        state_computation_config = dict(
            persistance=0.8, 
            sector_angle=45,
            berryworth_offset=0.05,
            max_berry_count = 800,
            noise=0.05
        ),
        exploration_subroutine_config = dict(
            reward_discount_factor=1.0
        ),
        reward_perception_config = dict(
            max_clip=3, min_clip=-0.04,
            scale=3 * 1e4/50
        ),
        nn_model_config = dict(
            layers=[32,16,16],
            lrelu_negative_slope=-0.01
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
        beta_rate=0.9/800
    ),

    "TRAINING_STRAT_EPSILON_GREEDY": dict(
        epsilon=0.55,
        finalepsilon=0.1,
        decaySteps=200,
        decay_type='exp'
    ),

    "DDQN": dict(
        batchSize=1024, 
        gamma=0.99, 
        update_freq=5, 
        MaxTrainEpisodes=500, 
        optimize_every_kth_action=100, #-1, 
        num_gradient_steps=25, #400,
        evalFreq=10, 
        printFreq=1, 
        polyak_average=True, 
        polyak_tau=0.1,
        gradient_clips = (-0.5,0.5),
    ),
}