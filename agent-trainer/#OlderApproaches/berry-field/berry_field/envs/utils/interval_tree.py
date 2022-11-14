import numpy as np

class Node:
    def __init__(self, x, y, nodeid:int) -> None:
        self.interval = (x,y)
        self.id = int(nodeid) if type(nodeid) != str else nodeid
        self.left = None
        self.right = None
        self.parent = None
        self.max = y
        self.min = x

class IntervalTree:
    def __init__(self, data, create_balanced_tree = True):
        # The data is assumed to be of the following form: [start, end, id],
        # where both start and end are included in the interval
        self.data = np.array(data)
        self.root = None
        self.map_nodeId_Node = {}
        if create_balanced_tree:
            self.create_balanced_tree(data)
        else:
            self.create_tree(data)


    def add_node(self, new_node: Node):
        node = self.root
        while(node is not None):
            if(new_node.interval[0] <= node.interval[0]):
                if(node.left is None):
                    node.left = new_node
                    new_node.parent = node
                    return
                node = node.left
            else:
                if(node.right is None):
                    node.right = new_node
                    new_node.parent = node
                    return
                node = node.right


    def update_MaxMinOfSubtree(self, subtree:Node):
        # runs for all nodes in subtree
        if subtree is None: return

        self.update_MaxMinOfSubtree(subtree.left)
        self.update_MaxMinOfSubtree(subtree.right)

        if subtree.left is not None:
            subtree.max = max(subtree.max, subtree.left.max)
            subtree.min = min(subtree.min, subtree.left.min)
        if subtree.right is not None:
            subtree.max = max(subtree.max, subtree.right.max)
            subtree.min = min(subtree.min, subtree.right.min)


    def create_tree(self, node_list):
        if len(node_list) == 0: return
        np.random.shuffle(node_list)
        x,y,nodeid = node_list[0]     
        self.root = Node(x,y,nodeid)   
        self.map_nodeId_Node[nodeid] = self.root 
        for x,y,nodeid in node_list[1:]:
            new_node = Node(x,y,nodeid)
            self.add_node(new_node)
            self.map_nodeId_Node[nodeid] = new_node
        self.update_MaxMinOfSubtree(self.root)


    def create_balanced_tree_helper(self, node_list, parent:Node):
        if len(node_list) == 0: return None
        mid = len(node_list)//2
        x,y,nodeid = node_list[mid] 
        new_node = Node(x,y,nodeid) 
        new_node.parent = parent
        new_node.left = self.create_balanced_tree_helper(node_list[:mid], new_node)
        new_node.right = self.create_balanced_tree_helper(node_list[mid+1:], new_node) 
        self.map_nodeId_Node[nodeid] = new_node
        return new_node


    def create_balanced_tree(self, node_list):
        if len(node_list) == 0: return    
        idxs = np.argsort(node_list[:,0], axis=0)
        node_list = node_list[idxs]
        mid = len(node_list)//2
        x,y,nodeid = node_list[mid] 
        self.root = Node(x,y,nodeid) 
        self.root.left = self.create_balanced_tree_helper(node_list[:mid], self.root)
        self.root.right = self.create_balanced_tree_helper(node_list[mid+1:], self.root)  
        self.map_nodeId_Node[nodeid] = self.root  
        self.update_MaxMinOfSubtree(self.root)


    def has_overlap(self, interval1:tuple, interval2:tuple):
        return interval1[0] <= interval2[1] and interval2[0] <= interval1[1] 


    def find(self, interval, root, result):
        if root is None: return
        if self.has_overlap(root.interval, interval):
            result.append(root.id)
        if root.left and self.has_overlap((root.left.min, root.left.max), interval):
            self.find(interval, root.left, result)
        if root.right and self.has_overlap((root.right.min, root.right.max), interval):
            self.find(interval, root.right, result)


    def find_overlaps(self, interval):
        result = []
        self.find(interval, self.root, result)
        return set(result)


    def floatup_MinMax(self, node: Node):
        """traverses to root while updating min-max of nodes in path"""
        if node is None: return

        node.min, node.max = node.interval
        if node.left is not None:
            node.min = min(node.min, node.left.min)
            node.max = max(node.max, node.left.max)
        if node.right is not None:
            node.min = min(node.min, node.right.min)
            node.max = max(node.max, node.right.max)
        self.floatup_MinMax(node.parent)


    def leftmost_node(self, subtree:Node):
        if subtree is None: return None
        node = subtree
        while(node.left is not None):
            node = node.left
        return node
    

    def delete_node(self, node_id):
        node = self.map_nodeId_Node[node_id]
        
        if node.left == None:
            if node.right is not None: node.right.parent = node.parent
            if node.parent is not None:
                if node.parent.left == node:
                    node.parent.left = node.right
                else:
                    node.parent.right = node.right
                self.floatup_MinMax(node.parent)
            else:
                self.root = node.right
                if self.root is not None: self.root.parent = None
            del node, self.map_nodeId_Node[node_id]
        
        elif node.right == None:
            if node.left is not None: node.left.parent = node.parent
            if node.parent is not None:
                if node.parent.left == node:
                    node.parent.left = node.left
                else:
                    node.parent.right = node.left
                self.floatup_MinMax(node.parent)
            else:
                self.root = node.left
                self.root.parent = None
            del node, self.map_nodeId_Node[node_id]
        
        else:
            successor_node = self.leftmost_node(node.right)
            node.id = successor_node.id
            node.interval = successor_node.interval
            self.delete_node(successor_node.id)
            self.map_nodeId_Node[node.id] = node
            del successor_node, self.map_nodeId_Node[node_id]




if __name__ == '__main__':
    data = [
        [20,400,'id01'],
        [30,300,'id02'],
        [500,700,'id03'],
        [1020,2400,'id04'],
        [29949, 35891,'id05'],
        [899999,900000,'id06'],
        [999000,999010,'id07']
    ]
    np.random.shuffle(data)
    tree = IntervalTree(data, True)


    print(tree.find_overlaps([100-10, 120+10]))
    print(tree.root.min, tree.root.max)
    tree.delete_node('id01')
    print(tree.find_overlaps([100-10, 120+10]))
    print(tree.root.min, tree.root.max)
