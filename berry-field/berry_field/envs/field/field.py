# this exposes some functions to interact with the berry field

from typing import Tuple
import numpy as np
from .collision_tree import collision_tree


class Field:
    def __init__(self, agent_size, berry_data) -> None:
        # berry_data is [[patch-no,size,x,y],...]
        self.AGENT_SIZE = agent_size
        self.berry_data = berry_data
        self.reset() # init with default params

    def reset(self, berry_data=None):
        if berry_data is None:
            self._make_trees(self.berry_data)
        else:
            self._make_trees(berry_data)

    def get_berries_in_view(
        self, bounding_box, return_ids=False
    ) -> Tuple[list, np.ndarray]:
        """ returns the bounding boxes of all the berries in the given 
        bounding box """
        boxIds, boxes = self.berry_collision_tree.boxes_within_overlap(
            bounding_box, return_boxes=True
        )
        if return_ids: return boxes, boxIds
        return boxes

    def pick_collided_berries(self, position):
        """pick the berries the agent collided with and return a reward
        make sure that self.position is correct/updated before calling this"""
        agent_bbox = (*position, self.AGENT_SIZE, self.AGENT_SIZE)
        boxIds, boxes = self.berry_collision_tree.find_collisions(
            agent_bbox, iscircle=True, radius=self.AGENT_SIZE/2, 
            return_boxes=True
        )
        self.berry_collision_tree.delete_boxes(list(boxIds))
        self.picked_berries = len(boxes)
        return boxes[:,:3] # [x,y,size]

    def get_current_patch(self, position):
        """ get the patch-id and bounding-box of the patch where the agent's 
        center is currently in and if the agent is in no patch, then it 
        returns None, None make sure that the postion is as intended before 
        calling this """
        pos_bbox = (*position, 0, 0) # represents a point
        overlaping_patches, boxes = self.patch_tree.boxes_within_overlap(
            pos_bbox, return_boxes=True
        )
        if len(overlaping_patches) > 0: 
            return overlaping_patches[0], boxes[0]
        return None, None

    def get_num_patches(self):
        return len(self.patch_tree.boxIds)

    def get_unique_berry_sizes(self):
        return np.unique(self.berry_collision_tree.boxes[:,2])
        
    def _create_bounding_boxes(self, berry_data):
        """ bounding boxes from berry-coordinates and 
        size:[centerx, centery, width, height] """
        bounding_boxes = np.column_stack([
            berry_data[:,2:], berry_data[:,1], berry_data[:,1]
        ])
        return bounding_boxes

    def _get_patch_boxes(self, berry_data):
        """ generate the bounding boxes for the patches by taking the 
        extreme berries it is assumed that berry data is of form 
        [[patch-no., size, x, y],...] """

        # get the rectangular enclosure for all the patches
        num_patches = len(np.unique(berry_data[:,0]))
        patch_rects = [
            [np.inf,0.0,np.inf,0.0] for patch_no in range(num_patches)
        ] # [left, right, bot, top]
        for berry in berry_data: # [patch-no, size, x, y]
            patch_id = int(berry[0])
            patch_rects[patch_id][0] = min(patch_rects[patch_id][0], berry[2]) # top x limit
            patch_rects[patch_id][1] = max(patch_rects[patch_id][1], berry[2]) # bot x limit
            patch_rects[patch_id][2] = min(patch_rects[patch_id][2], berry[3]) # top y limit
            patch_rects[patch_id][3] = max(patch_rects[patch_id][3], berry[3]) # bot y limit

        # convert rects to bounding boxes
        for i in range(num_patches):
            left, right, bot, top = patch_rects[i]
            centerx, centery = (left + right)/2, (bot + top)/2
            width = right - left
            height = top - bot
            patch_rects[i] = [centerx, centery, width, height]
        patch_bboxes = np.array(patch_rects)

        # padding to assure that berries can be collected only inside a patch
        max_berry_size = max(berry_data[:,1])
        patch_bboxes[:,2] += max_berry_size + self.AGENT_SIZE
        patch_bboxes[:,3] += max_berry_size + self.AGENT_SIZE

        return patch_bboxes

    def _make_trees(self, berry_data: np.ndarray):

        berry_radii = berry_data[:,1]/2 # size the diameter of berries
        # bounding_boxes are [x,y,width,height]
        bounding_boxes = self._create_bounding_boxes(berry_data) 
        
        # compute patch boundaries using berry data
        patch_boxes = self._get_patch_boxes(berry_data)

        # make the berry and patch collision trees
        self.patch_tree = collision_tree(patch_boxes)
        self.berry_collision_tree = collision_tree(
            bounding_boxes, circle_in_box=True, radii=berry_radii
        )