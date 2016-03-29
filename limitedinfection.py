#!/usr/bin/env python3.5


"""
Khan Academy's Limited Infection Problem
========================================
Some notes:
    * Some directed edge from A->B indicates that A coaches B.
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def main():
    infection = NetworkInfection()
    infection.load()
    infection.choose()
    infection.show()

class NetworkInfection(object):
    def __init__(self, filename='./testnetwork.npy', refresh=False, choose_node=False):
        self.networkfile = filename
        self.graph       = None
        self.nxgraph     = None
        self.choice      = choose_node

        if refresh:
            self._gen_new_random_graph()
            self.filename = './testnetwork.npy'

    def load(self):
        self.graph   = np.load(self.networkfile)
        self.nxgraph = nx.DiGraph(self.graph)

    def show(self):
        plt.figure()
        nx.draw(self.nxgraph, pos=nx.spring_layout(self.nxgraph))
        plt.show()

    def _gen_new_random_graph(self):
        newgraph = nx.binomial_graph(20, 0.1, directed=True)
        np.save('testnetwork.npy', nx.adjacency_matrix(newgraph).todense())

    def choose(self):
        """
        TODO: Need to make sure choice is part of connected subgraph. (not trivial)
        """
        if type(self.choice) == bool:
            if self.choice:
                self.choice = input('Select Node')
            else:
                self.choice = np.random.choice(self.nxgraph.nodes())

    def total_infection(self):
        pass

    def limited_infection(self):
        pass



if __name__ == '__main__':
    sys.exit(main())
