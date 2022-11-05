# this will be a summary of what ever constant argument is required
CONFIG = {

    "RND_TRAIN_ENV": dict(
        field_size=(20000,20000), 
        patch_size=(2600,2600), 
        num_patches=10, seperation=2400, 
        nberries=80, spawn_radius=100, 
        initial_juice=0.5,
        maxtime=5*60, play_till_maxtime = True,
        patch_with_agent_at_center=True,
        penalize_boundary_hit=False
    ),

    "AGENT": dict(
        # params controlling the state and state-transitions
        angle = 45, persistence=0.8, worth_offset=0.05, 
        noise=0.01, nstep_transition=[1], positive_emphasis=0,
        skipStep=10, patch_discovery_reward=0.0, 
        add_exploration = True,
        reward_magnification = 3 * 1e4/50,
        perceptable_reward_range = [-0.04,3],

        # params related to time memory
        time_memory_factor=0.6, time_memory_exp=1.0,
        time_memory_grid_sizes= [
            (20,20),(50,50),(100,100),(200,200),(400,400)
        ],

        # # params related to berry memory
        # berry_memory_grid_size = (400,400),

        # other params
        render=False, 
        debug=False, debugDir='.temp',
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
        gamma=0.9, 
        update_freq=5, 
        MaxTrainEpisodes=800, 
        optimize_every_kth_action=100, #-1, 
        num_gradient_steps=25, #400,
        evalFreq=10, 
        printFreq=1, 
        polyak_average=True, 
        polyak_tau=0.1,
        resumeable_snapshot=10,
        gradient_clips = (-0.5,0.5),
    ),

    "WANDB": dict(
        ENABLE_WANDB = True, # set to true for server env
        project="Agent-Design",
        group="tuning-berry-picked-bool-feature-2.1.2",
        entity="foraging-rl",
        watch_log_freq = 100,
    ),

    "seed": 4, # seed for random, np.random, torch
    "LOG_DIR_PARENT": ".temp/tuning-berry-picked-bool-feature-2.1.2", # the log folder for all runs
    "RESUME_DIR": None, # set if resuming a run
}