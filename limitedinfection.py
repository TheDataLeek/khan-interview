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
import matplotlib.animation as animation
import networkx as nx


def main():
    infection = NetworkInfection()
    infection.load()
    infection.choose()
    infection.total_infection(animate=True)

class NetworkInfection(object):
    def __init__(self, filename='./testnetwork.npy', refresh=False, choose_node=False):
        self.networkfile = filename
        self.graph       = None
        self.nxgraph     = None
        self.choice      = choose_node
        self.infections  = None

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
        newgraph = nx.binomial_graph(30, 0.2, directed=True)
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
        self._infection_list()

    def _infection_list(self):
        self.infections = {n:(True if n == self.choice else False) for n in self.nxgraph.nodes()}

    def total_infection(self, animate=False):
        """
        This part is straightforward, just simple graph traversal.

        Initially we will assume heavily connected graph (no independent subgraphs)

        TODO: Deal with independent subgraphs (own assertion probably)
        """
        inf_sort = lambda l: sorted(l, key=lambda tup: tup[0])
        states = [inf_sort(self.infections.items())]
        bfs = nx.bfs_edges(self.nxgraph, self.choice)   # DFS would also work here
        for start, end in bfs:
            self.infections[end] = True
            states.append(inf_sort(self.infections.items()))
        if animate:
            self.animate_infection(states)

    def animate_infection(self, states):
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        pos = nx.spring_layout(self.nxgraph)
        colors = [0 if infection is False else 1 for node, infection in states[0]]
        nodes = nx.draw_networkx_nodes(self.nxgraph, pos=pos, node_color=colors)
        edges = nx.draw_networkx_edges(self.nxgraph, pos=pos)

        def animate(i):
            colors = [0 if infection is False else 1 for node, infection in states[i]]
            nodes = nx.draw_networkx_nodes(self.nxgraph, pos=pos, node_color=colors)
            return nodes, edges

        def init():
            return nodes, edges

        ani = animation.FuncAnimation(fig, animate, np.arange(len(states)), init_func=init,
                interval=20)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Will Farmer'), bitrate=1800)
        ani.save('infection.mp4', writer=writer)

        plt.show()

    def limited_infection(self):
        pass



if __name__ == '__main__':
    sys.exit(main())
