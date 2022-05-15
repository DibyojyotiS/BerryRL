from berry_field.envs.berry_field_env import BerryFieldEnv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

BERRY_SIZES = [10,20,30,40]
AGENT_SIZE=10

def random_baby_berryfield(field_size=(4000,4000), patch_size = (1000,1000), 
                        num_patches=5, n_berries=10, show=False):
    """ n_berries: number of berries per patch """
    max_berry_size = max(BERRY_SIZES)
    pw,ph = patch_size

    # make random patches that don't overlap
    offset_x, offset_y = pw/2+max_berry_size, ph/2+max_berry_size
    low_x, high_x = offset_x, field_size[0]-offset_x
    low_y, high_y = offset_y, field_size[1]-offset_y
    while True:
        patch_centers_x = np.random.randint(low_x, high_x, num_patches)
        patch_centers_y = np.random.randint(low_y, high_y, num_patches)
        collision_found = False
        for i in range(num_patches):
            for j in range(i+1, num_patches):
                x1,y1 = patch_centers_x[i],patch_centers_y[i]
                x2,y2 = patch_centers_x[j],patch_centers_y[j]
                if abs(x1-x2) < pw and abs(y1-y2) < ph: 
                    collision_found = True
                    break
        if not collision_found: break

    # generate the berries
    berry_data = np.zeros((num_patches*n_berries, 4))
    for i,(px,py) in enumerate(zip(patch_centers_x, patch_centers_y)):
        berry_data[i*n_berries:(i+1)*n_berries, 0] = i
        berry_data[i*n_berries:(i+1)*n_berries, 1] = np.random.choice(BERRY_SIZES, n_berries)
        low_x, low_y = 2*max_berry_size - pw/2, 2*max_berry_size - ph/2,
        high_x, high_y = pw/2-2*max_berry_size, ph/2-2*max_berry_size
        berry_data[i*n_berries:(i+1)*n_berries, 2] = np.random.randint(low_x, high_x, n_berries) + px
        berry_data[i*n_berries:(i+1)*n_berries, 3] = np.random.randint(low_y, high_y, n_berries) + py

    # set the initial position
    dist_to_keep = berry_data[:,1] + AGENT_SIZE
    while True:
        i = np.random.randint(0, num_patches)
        init_x = np.random.randint(0, pw) - pw/2 + patch_centers_x[i]
        init_y = np.random.randint(0, ph) - ph/2 + patch_centers_y[i]
        dists = np.abs(berry_data[:,2:] - [init_x,init_y])
        no_collisions = True
        for (dx, dy), dk in zip(dists, dist_to_keep):
            if dx < dk and dy < dk:
                no_collisions = False; break
        if no_collisions: break
    initial_position = (init_x,init_y)

    if show:
        fig, ax = plt.subplots()
        for x,y in zip(patch_centers_x, patch_centers_y): 
            ax.add_patch(Rectangle((x-pw/2,y-ph/2), *patch_size, fill=False))
        ax.add_patch(Rectangle((0,0), *field_size, fill=False))
        ax.scatter(x = berry_data[:,2], y=berry_data[:,3], 
                    s=berry_data[:,1], c='r', zorder=num_patches+1)
        ax.scatter(x=init_x, y=init_y, c='black', s=10)
        plt.show()

    return berry_data, initial_position


def getBabyEnv(field_size=(4000,4000), patch_size=(1000,1000), num_patches=5, nberries=10, 
                logDir='.temp', living_cost=True, initial_juice=0.5, end_on_boundary_hit= False, 
                penalize_boundary_hit=False, show=False):
    # making the berry env
    random_berry_data, random_init_pos = random_baby_berryfield(field_size, patch_size, 
                                                                num_patches, nberries, show)
    berry_env = BerryFieldEnv(no_action_r_threshold=float('inf'),
                                field_size=field_size,
                                initial_position=random_init_pos,
                                user_berry_data= random_berry_data,
                                end_on_boundary_hit= end_on_boundary_hit,
                                penalize_boundary_hit= penalize_boundary_hit,
                                initial_juice= initial_juice,
                                analytics_folder=logDir,
                                enable_analytics = logDir is not None)

    # redefine the reset function to generate random berry-envs
    def env_reset(berry_env_reset):
        def reset(**args):
            berry_data, initial_pos = random_baby_berryfield(field_size, patch_size, 
                                                            num_patches, nberries, show) # reset the env  
            x = berry_env_reset(berry_data=berry_data, initial_position=initial_pos)
            return x
        return reset

    def env_step(berry_env_step):
        if living_cost: 
            print('with living cost, rewards scaled by 1/(berry_env.REWARD_RATE*MAXSIZE)')
        else: print('no living cost, rewards (except boundary hit) scaled by 1/(berry_env.REWARD_RATE*MAXSIZE)')
        MAXSIZE = max(BERRY_SIZES)
        scale = 1/(berry_env.REWARD_RATE*MAXSIZE)
        def step(action):
            state, reward, done, info = berry_env_step(action)
            if living_cost: reward = scale*reward
            else: reward = (scale*(reward>0) + (reward<=-1))*reward # no living cost
            return state, reward, done, info
        return step

    berry_env.reset = env_reset(berry_env.reset)
    berry_env.step = env_step(berry_env.step)

    return berry_env