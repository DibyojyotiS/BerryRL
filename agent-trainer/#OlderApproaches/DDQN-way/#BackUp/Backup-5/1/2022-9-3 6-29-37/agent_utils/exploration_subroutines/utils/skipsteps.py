def skip_steps(action, skipSteps, berryenv_step):
    """ repeats the same action for skipSteps+1 number of steps
    ### parameters
    1. action: int
    2. skipSteps: int
    3. berryenv_step: function (int) -> listberries, reward, done, info
            - the step function from an instance of BerryFieldEnv

    ### return
    - sum_reward: the summation of reward while skipping frames
    - skip_trajectory: a list of [listberries, info, reward, done]
                        in accordance to DRLagents' convention
    - steps: the total steps actually taken
    """
    sum_reward = 0; skip_trajectory = []
    for steps in range(skipSteps+1):
        listberries, reward, done, info = berryenv_step(action)
        sum_reward += reward
        skip_trajectory.append([listberries, info, reward, done])
        if done: break
    return sum_reward, skip_trajectory, steps+1
