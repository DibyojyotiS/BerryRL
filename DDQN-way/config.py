# this will be a summary of what ever constant argument is required
CONFIG = {

    "RND_TRAIN_ENV": dict(
        field_size=(20000,20000), 
        patch_size=(2600,2600), 
        num_patches=10, seperation=2400, 
        nberries=80, spawn_radius=100, 
        initial_juice=0.5,
        patch_with_agent_at_center=True,
        penalize_boundary_hit=False
    ),

    "AGENT": dict(
        # params controlling the state and state-transitions
        angle = 45, persistence=0.8, worth_offset=0.05, 
        noise=0.01, nstep_transition=[1], positive_emphasis=0,
        skipStep=10, reward_patch_discovery=True, 
        add_exploration = True, spacings=[],

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
        lr=0.0001, weight_decay=0.0
    ),

    "MULTI_STEP_LR": dict(
        milestones=[50*i for i in range(1,21)],
        gamma=0.5
    ),

    "PER_BUFFER": dict(
        bufferSize=int(6E4), 
        alpha=0.95,
        beta=0.1, 
        beta_rate=0.9/2000
    ),

    "TRAINING_STRAT_EPSILON_GREEDY": dict(
        epsilon=0.5,
        finalepsilon=0.2,
        decaySteps=1000
    ),

    "DDQN": dict(
        batchSize=512, 
        gamma=0.9, 
        update_freq=5, 
        MaxTrainEpisodes=2000, 
        optimize_every_kth_action=100, 
        num_gradient_steps=25,
        evalFreq=10, 
        printFreq=1, 
        polyak_average=True, 
        polyak_tau=0.1,
        resumeable_snapshot=10,
    ),

    "WANDB": dict(
        ENABLE_WANDB = True, # set to true for server env
        project="Agent-Design",
        group="multi-resolution-time-memory",
        entity="foraging-rl",
        watch_log_freq = 100,
    ),

    "seed": 0, # seed for random, np.random, torch
    "LOG_DIR_PARENT": ".temp", # the log folder for all runs
    "RESUME_DIR": None, # set if resuming a run
}