import numpy as np
import gym

env = gym.make('berry_field:berry_field_mat_input-v0')
bounding_boxes = env.berry_collision_tree.boxes
berry_collision_tree = env.berry_collision_tree

print("check positions of berries")
found_Ids = []
not_found_atpos = []
for i, berry in enumerate(bounding_boxes):
    x,y,_,_ = berry
    berry_point = (x,y, 0.1, 0.1)
    berryIds = berry_collision_tree.find_collisions(berry_point, True, 0.05)
    if i not in berryIds: not_found_atpos.append(i)
    for berryId in berryIds: found_Ids.append(berryId)
found_Ids = [*set(found_Ids)]
print("missplaced: ", not_found_atpos)
print("notfound:", set(np.arange(bounding_boxes.shape[0])) - set(found_Ids))
print("not in X", set([*berry_collision_tree.intervaltreeX.map_nodeId_Node.keys()]) - set(found_Ids))
print("not in Y", set([*berry_collision_tree.intervaltreeY.map_nodeId_Node.keys()]) - set(found_Ids))
print("if the above are empty sets then all berries are there")


def f(root, mode='x'):
    m = {i:0 for i in range(0,800)}
    def inord(root):
        if root is None: return
        inord(root.left)
        m[root.id]+=1
        inord(root.right)
    inord(root)
    b = False
    for x in m.keys():
        if m[x]>1:
            if not b: print(f'\t{mode} ', end='')
            b=True
            print(f"{x}:{m[x]}", end=' ')
    if b: print()


print("check berry deletion")
np.random.shuffle(found_Ids)
for i, berryId in enumerate(found_Ids):
    try:
        berry_collision_tree.delete_boxes([int(berryId)])
    except KeyError:
        print("KeyError: ", berryId)
    f(env.berry_collision_tree.intervaltreeX.root, 'x')
    f(env.berry_collision_tree.intervaltreeY.root, 'y')
    temp = []
    for avId in found_Ids[i+1:]:
        x,y,_,_ = bounding_boxes[avId]
        berry_point = (x,y, 0.1, 0.1)
        berryIds = berry_collision_tree.find_collisions(berry_point, True, 0.05)
        if avId not in berryIds: temp.append(avId)
    if len(temp)>0:print(i,temp)
print("if there are no outputs after 'check berry deletion' then deletion is working")
