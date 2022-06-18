class PatchDiscoveryReward:
    def __init__(self,reward_value=1.0):
        self.reward = reward_value
        self.visited_patches = set()
    
    def __call__(self, info)->float:
        """ call with info=None to reset """
        # None means the start of new episode
        # reset the visited_patches to empty set
        if info is None: 
            self.visited_patches = set()
            return 0

        patch_id = info['current-patch-id']
        if patch_id is None or \
            patch_id in self.visited_patches: 
            return 0

        # don't give reward to spawn patch!!!
        self.visited_patches.add(patch_id)
        return self.reward * (len(self.visited_patches) > 1)