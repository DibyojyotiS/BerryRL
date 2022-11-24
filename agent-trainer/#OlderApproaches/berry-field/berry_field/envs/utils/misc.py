from functools import cmp_to_key
import numpy as np


def argsort_clockwise(directions):
    tmepdirections = np.column_stack([np.arange(directions.shape[0]), directions])
    cmp = lambda x,y: isClockwise(x[1:], y[1:])
    args = np.array(sorted(tmepdirections, key=cmp_to_key(cmp)))[:,0]
    return args.astype(int)


def isClockwisehelper(v):
    # partitions circle into two sub spaces and returns True/False depending where v is
    rx,ry = (0,1) #reference vector
    x,y = v
    curl = rx*y - ry*x
    dot = x*rx + y*ry
    if curl < 0: return True
    if curl == 0 and dot == 1: return True
    return False


def isClockwise(v1, v2):
    x1,y1,x2,y2=(*v1,*v2)
    curl = x1*y2 - x2*y1
    # dot = x1*x2 + y1*y2
    v1_in_A = isClockwisehelper(v1)
    v2_in_A = isClockwisehelper(v2)
    if(v1_in_A == v2_in_A):
        if(curl < 0): return -1
        return 1
    elif(v1_in_A and not v2_in_A):
        return -1
    return 1


def getTrueAngles(directions, referenceVector=[0,1]):
    curls = np.cross(directions, referenceVector)
    dot = np.dot(directions, referenceVector)
    angles = np.arccos(dot)*180/np.pi
    args0 = np.argwhere(np.bitwise_not((curls > 0)|(curls == 0)&(dot==1)))
    angles[args0] = 360-angles[args0]
    return angles