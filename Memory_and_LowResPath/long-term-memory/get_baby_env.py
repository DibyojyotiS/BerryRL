from berry_field.envs.berry_field_env import BerryFieldEnv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from print_utils import printLocals

BERRY_SIZES = [10,20,30,40]
AGENT_SIZE=10


def _fill_berries(num_patches, n_berries, patch_centers_x, patch_centers_y, patch_size):
    pw,ph = patch_size
    max_berry_size = max(BERRY_SIZES)
    # generate the berries [patch#, size, x,y]
    berry_data = np.zeros((num_patches*n_berries, 4))
    for i,(px,py) in enumerate(zip(patch_centers_x, patch_centers_y)):
        berry_data[i*n_berries:(i+1)*n_berries, 0] = i
        berry_data[i*n_berries:(i+1)*n_berries, 1] = np.random.choice(BERRY_SIZES, n_berries)
        low_x, low_y = 2*max_berry_size - pw/2, 2*max_berry_size - ph/2,
        high_x, high_y = pw/2-2*max_berry_size, ph/2-2*max_berry_size
        berry_data[i*n_berries:(i+1)*n_berries, 2] = np.random.randint(low_x, high_x, n_berries) + px
        berry_data[i*n_berries:(i+1)*n_berries, 3] = np.random.randint(low_y, high_y, n_berries) + py
    return berry_data


def _IsSeperatedNoOverlap(patch_size, separation, patch_centers_x, patch_centers_y):
    pw, ph = patch_size
    num_patches = len(patch_centers_x)
    for i in range(num_patches):
        for j in range(i+1, num_patches):
            x1,y1 = patch_centers_x[i],patch_centers_y[i]
            x2,y2 = patch_centers_x[j],patch_centers_y[j]
            if abs(x1-x2) < pw+separation and abs(y1-y2) < ph+separation: 
                return False
    return True


def _rejection_sample(patch_size, max_berry_size, field_size, 
                    seperation, num_patches, n_berries):
    pw,ph = patch_size

    # make random patches that don't overlap and are seperated by atleast seperation
    offset_x, offset_y = pw/2+max_berry_size, ph/2+max_berry_size
    low_x, high_x = offset_x, field_size[0]-offset_x
    low_y, high_y = offset_y, field_size[1]-offset_y
    while True:
        patch_centers_x = np.random.randint(low_x, high_x, num_patches)
        patch_centers_y = np.random.randint(low_y, high_y, num_patches)
        if _IsSeperatedNoOverlap(num_patches, patch_size, 
            seperation, patch_centers_x, patch_centers_y):
            break

    berry_data = _fill_berries(num_patches, n_berries, 
                    patch_centers_x, patch_centers_y, patch_size)

    return berry_data, patch_centers_x, patch_centers_y


def _incremental_sample(patch_size, max_berry_size, field_size, 
                    seperation, num_patches, n_berries):
    """ incrementally build the berry-field. Patch centers are sampled
    one-by-one from a uniform distribution over the field-size, and appeded
    to the patch-list if they do-not overlap and are seperated by seperation. 
    If incremental construction is stuck then reject and repeat """
    pw,ph = patch_size
    num_tries = 10
    # make random patches that don't overlap and are seperated by atleast seperation
    offset_x, offset_y = pw/2+max_berry_size, ph/2+max_berry_size
    low_x, high_x = offset_x, field_size[0]-offset_x
    low_y, high_y = offset_y, field_size[1]-offset_y
    patch_centers_x = []; patch_centers_y = []
    while len(patch_centers_x)!=num_patches:
        for i in range(num_tries):
            patch_centers_x.append(np.random.uniform(low_x, high_x))
            patch_centers_y.append(np.random.uniform(low_y, high_y))
            passed = _IsSeperatedNoOverlap(patch_size, seperation, 
                                    patch_centers_x, patch_centers_y)
            if passed: break # patch remains in patch-list
            patch_centers_x.pop()
            patch_centers_y.pop()
        # if possibly stuck then repeate
        if not passed: 
            patch_centers_x = []
            patch_centers_y = []

    patch_centers_x = np.array(patch_centers_x)
    patch_centers_y = np.array(patch_centers_y)
    berry_data = _fill_berries(num_patches, n_berries, 
                    patch_centers_x, patch_centers_y, patch_size)

    return berry_data, patch_centers_x, patch_centers_y


def _random_initial_position_in_a_patch(num_patches, n_berries, patch_size, 
                                    berry_data, patch_centers_x, patch_centers_y):
    """ select a random initial_position in a random patch. Agent of 
    AGENT_SIZE at initial_position will not intersect any berry """
    pw,ph = patch_size
    dist_to_keep = (berry_data[:,1] + AGENT_SIZE)/2 + 1
    while True:
        patchIdx = np.random.randint(0, num_patches)
        init_x = int(np.random.randint(0, pw) - pw/2 + patch_centers_x[patchIdx])
        init_y = int(np.random.randint(0, ph) - ph/2 + patch_centers_y[patchIdx])
        patch_berries = berry_data[patchIdx*n_berries:(patchIdx+1)*n_berries]
        patch_dist_to_keep = dist_to_keep[patchIdx*n_berries:(patchIdx+1)*n_berries]
        patch_dists = np.abs(patch_berries[:,2:] - [init_x,init_y])
        no_collisions = True
        for (dx, dy), dk in zip(patch_dists, patch_dist_to_keep):
            if dx < dk and dy < dk:
                no_collisions = False; break
        if no_collisions: break
    initial_position = (init_x,init_y) 
    return initial_position


def _random_initial_position_around_a_berry(berry_data, field_size, spawn_radius):
    """ select an random initial_position about a random berry """
    collision = True
    while collision:
        rndIndx = np.random.randint(0, len(berry_data))
        _, size, x, y = berry_data[rndIndx]
        rndR = np.random.randint((size+AGENT_SIZE)/2 + 1, spawn_radius) # 1 added just for sake of it
        rndAngle = np.random.uniform(0, 2*np.pi)
        initial_position = int(x + rndR*np.cos(rndAngle)), int(y + rndR*np.sin(rndAngle))

        # if agent intersects with boundary then try again
        if (initial_position[0]-AGENT_SIZE/2 <= 0) or (initial_position[1]-AGENT_SIZE/2<=0) \
            or (initial_position[0]+AGENT_SIZE/2 >= field_size[0]) \
            or (initial_position[1]+AGENT_SIZE/2 >= field_size[1]): continue

        dists = np.linalg.norm(berry_data[:,2:]-initial_position)
        collision = any(dists < (berry_data[:,1]+AGENT_SIZE)/2 + 1)
    return initial_position


def random_baby_berryfield(field_size=(20000,20000), patch_size = (2600,2600), 
                        num_patches=10, seperation=5000, n_berries=80, 
                        initial_pos_around_berry = True, spawn_radius=100, 
                        sampling_type=0, show=False):
    """ n_berries: number of berries per patch 
    initial_pos_around_berry: if True agent will start within spawn_radius about a  
                            random berry else at a random position inside a random patch
    sampling_type: if set to 0 then the patches berry-field is sampled incrementally (fast),
                    other wise all patches are sampled randomly-uniformly and then checked 
                    for overlaps and sepearation (slower for less solutions).
    """
    max_berry_size = max(BERRY_SIZES)

    # make the berry-field
    args = [patch_size, max_berry_size, field_size, 
            seperation, num_patches, n_berries]

    if sampling_type == 0:
        berry_data, patch_centers_x, patch_centers_y = _incremental_sample(*args)
    else:
        berry_data, patch_centers_x, patch_centers_y = _rejection_sample(*args)

    # set the initial position
    if initial_pos_around_berry:
        initial_position = _random_initial_position_around_a_berry(berry_data, field_size, spawn_radius)
    else:
        initial_position = _random_initial_position_in_a_patch(num_patches,n_berries,
                                patch_size,berry_data, patch_centers_x, patch_centers_y)    

    if show:
        fig, ax = plt.subplots()
        for x,y in zip(patch_centers_x-patch_size[0]/2, patch_centers_y-patch_size[1]/2): 
            ax.add_patch(Rectangle((x,y), *patch_size, fill=False))
        ax.add_patch(Rectangle((0,0), *field_size, fill=False))
        ax.scatter(x = berry_data[:,2], y=berry_data[:,3], 
                    s=berry_data[:,1], c='r', zorder=num_patches+1)
        ax.scatter(x=initial_position[0], y=initial_position[1], c='black', s=10)
        plt.show()

    return berry_data, initial_position


def getBabyEnv(field_size=(20000,20000), patch_size=(2600,2600), num_patches=10, nberries=80, 
                logDir=None, living_cost=True, initial_juice=0.5, end_on_boundary_hit= False, 
                penalize_boundary_hit=False, initial_pos_around_berry = True, spawn_radius=100,
                seperate_patches_by=5000, allow_no_action=False, no_action_threshold=0.7, 
                show=False):
    printLocals('getBabyEnv', locals())
    # making the berry env
    random_berry_data, random_init_pos = random_baby_berryfield(field_size, patch_size, 
                                            num_patches, nberries, initial_pos_around_berry, 
                                            spawn_radius, seperate_patches_by, show=show)
    berry_env = BerryFieldEnv(noAction_juice_threshold=no_action_threshold,
                                field_size=field_size,
                                initial_position=random_init_pos,
                                user_berry_data= random_berry_data,
                                end_on_boundary_hit= end_on_boundary_hit,
                                penalize_boundary_hit= penalize_boundary_hit,
                                initial_juice= initial_juice,
                                allow_action_noAction=allow_no_action,
                                analytics_folder=logDir,
                                enable_analytics = logDir is not None)

    # redefine the reset function to generate random berry-envs
    def env_reset(berry_env_reset):
        def reset(**args):
            berry_data, initial_pos = random_baby_berryfield(field_size, patch_size, 
                                        num_patches, nberries, initial_pos_around_berry, 
                                        spawn_radius, seperate_patches_by, show=show) # reset the env  
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