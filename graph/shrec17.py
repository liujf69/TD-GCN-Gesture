import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 22 
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (3, 1), (4, 3), (5, 4), (6, 5), (7, 2), (8, 7), (9, 8), (10, 9), (11, 2), (12, 11),
                     (13, 12), (14, 13), (15, 2), (16, 15), (17, 16), (18, 17), (19, 2), (20, 19), (21, 20), (22, 21),
                     (2, 2)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index] 
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

