from typing import Tuple, Union
import numpy as np
from .interval_tree import IntervalTree

class collision_tree():
    def __init__(self, bounding_boxes:np.ndarray, boxIds, circle_in_box=False, radii=None) -> None:
        """ bounding_boxes array of [centerx, centery, width, height] denoting rectangles
            circle_in_box: if True the center is taken as the center of the box
            diameters: np.array, required if circle_in_box is true
            returns the corresponding boxIds on find_collisions"""

        assert bounding_boxes.ndim == 2 and bounding_boxes.shape[1] == 4

        self.num_rectangles = bounding_boxes.shape[0]
        self.boxes = bounding_boxes
        self.boxIds = boxIds

        _dataX = np.column_stack([bounding_boxes[:,0] - bounding_boxes[:,2]/2, 
                                bounding_boxes[:,0] + bounding_boxes[:,2]/2, self.boxIds])

        _dataY = np.column_stack([bounding_boxes[:,1] - bounding_boxes[:,3]/2,
                                bounding_boxes[:,1] + bounding_boxes[:,3]/2, self.boxIds])

        self.intervaltreeX = IntervalTree(_dataX)
        self.intervaltreeY = IntervalTree(_dataY)

        if circle_in_box: assert radii is not None
        self.circle_in_box = circle_in_box
        self.radii = radii


    def circle_rectangle_collisions(self, circle, rectangle):
        """ collisions between circle [x, y, r] and rectangle [x,y,width,height] """
        distx = abs(circle[0] - rectangle[0])
        disty = abs(circle[1] - rectangle[1])

        # no intersection
        if(distx > rectangle[2]/2 + circle[2]): return False
        if(disty > rectangle[3]/2 + circle[2]): return False
        
        # overlap along edges of rectangle
        if(distx <= rectangle[2]/2): return True
        if(disty <= rectangle[3]/2): return True

        # rectangle's corner in circle
        corner_dist = (distx - rectangle[2]/2)**2 + (disty - rectangle[3]/2)**2
        if corner_dist <= circle[2]**2: return True
        
        return False

    
    def circle_circle_collisions(self, circle1, circle2):
        """ collisions between circles [x, y, r] """
        dist = (circle1[0] - circle2[0])**2 + (circle1[1] - circle2[1])**2
        return dist <= (circle1[2]+circle2[2])**2


    def boxes_within_bound(self, bounding_box, return_boxes=False) -> Union[set, Tuple[set, np.ndarray]]:
        """boxIds for boxes overlapping with bounding box
            also returns the boxes [x,y,width,height] if return_boxes=True """
        intervalx = (bounding_box[0] - bounding_box[2]/2, bounding_box[0] + bounding_box[2]/2)
        intervaly = (bounding_box[1] - bounding_box[3]/2, bounding_box[1] + bounding_box[3]/2)

        overlapsx = self.intervaltreeX.find_overlaps(intervalx)
        overlapsy = self.intervaltreeY.find_overlaps(intervaly)
        overlapIds = overlapsx.intersection(overlapsy)

        if return_boxes:
            return overlapIds, self.boxes[list(overlapIds)]

        return overlapIds


    def find_collisions(self, bounding_box, iscircle=False, radius=None, return_boxes=False):
        """ collision detection with bounding_box [x,y,width,height] and other
            radi: float, required if iscircle = True
            iscircle: find collision with circle at center of bounding_box"""

        # both rect
        overlapIds = self.boxes_within_bound(bounding_box)

        # both circle
        if self.circle_in_box and iscircle:
            AgentCircle = (bounding_box[0], bounding_box[1], radius)
            for Id in list(overlapIds):
                circle2 = (self.boxes[Id][0], self.boxes[Id][1], self.radii[Id])
                collision = self.circle_circle_collisions(AgentCircle, circle2)
                if not collision: overlapIds.remove(Id)
            if return_boxes: return overlapIds, self.boxes[list(overlapIds)]
            return overlapIds
        
        # circle berry rect agent
        if self.circle_in_box and not iscircle:
            for Id in list(overlapIds):
                circle = (self.boxes[Id][0], self.boxes[Id][1], self.radii[Id])
                collision = self.circle_rectangle_collisions(circle, bounding_box)
                if not collision: overlapIds.remove(Id)
            if return_boxes: return overlapIds, self.boxes[list(overlapIds)]
            return overlapIds

        # rect berry circle agent
        if not self.circle_in_box and iscircle:
            AgentCircle = (bounding_box[0], bounding_box[1], radius)
            for Id in list(overlapIds):
                berry_box = self.boxes[Id]
                collision = self.circle_rectangle_collisions(AgentCircle, berry_box)
                if not collision: overlapIds.remove(Id)
            if return_boxes: return overlapIds, self.boxes[list(overlapIds)]
            return overlapIds 

        if return_boxes: return overlapIds, self.boxes[list(overlapIds)]
        return overlapIds


    def delete_boxes(self, boxIds:list):
        for boxId in boxIds:
            self.intervaltreeX.delete_node(boxId)
            self.intervaltreeY.delete_node(boxId)


    def get_box(self, boxId:int):
        return self.boxes[boxId]