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
import argparse

# I have a big screen
np.set_printoptions(linewidth=160)


def main():
    args = get_args()

    infection = NetworkInfection(args.nodes, args.prob, args.write, refresh=args.refresh)
    infection.load()
    infection.choose()
    states = infection.limited_infection()
    if args.animate:
        infection.animate_infection(states)

class NetworkInfection(object):
    def __init__(self, nodecount, prob, write, filename='./testnetwork.npy', refresh=False, choose_node=False):
        self.networkfile = filename
        self.graph       = None
        self.nxgraph     = None
        self.choice      = choose_node
        self.write       = write
        self.infections  = None
        self.subgraphs   = False

        if refresh:
            self._gen_new_random_graph(nodecount, prob)
            self.filename = './testnetwork.npy'

    def load(self):
        self.graph   = np.load(self.networkfile)
        self.nxgraph = nx.DiGraph(self.graph)
        if nx.number_weakly_connected_components(self.nxgraph) > 1:
            self.subgraphs = True

    def show(self):
        plt.figure()
        nx.draw(self.nxgraph, pos=nx.spring_layout(self.nxgraph))
        plt.show()

    def _gen_new_random_graph(self, nodecount, prob):
        newgraph = nx.binomial_graph(nodecount, prob)
        #np.save('testnetwork.npy', nx.adjacency_matrix(newgraph).todense())

    def choose(self):
        """
        TODO: Need to make sure choice is part of connected subgraph. (not trivial)
        """
        if type(self.choice) == bool:   # Prevent from re-picking
            if self.choice:
                self.choice = [input('Select Node(s)')]   # Not really intended for use
            else:
                self.choice = []
                for g in nx.weakly_connected_component_subgraphs(self.nxgraph):
                    self.choice.append(np.random.choice(g.nodes()))
        self._infection_list()

    def _infection_list(self):
        self.infections = {n:(True if n in self.choice else False) for n in self.nxgraph.nodes()}

    def _infection_sort(self, inf_list):
        """
        Need to sort list form of infection as dictionaries are unsorted
        """
        return sorted(inf_list, key=lambda tup: tup[0])

    def total_infection(self):
        """
        This part is straightforward, just simple graph traversal. (on connected subgraphs)
        """
        states = [self._infection_sort(self.infections.items())]

        subgraphs = list(nx.weakly_connected_component_subgraphs(self.nxgraph))
        for i in range(len(subgraphs)):
            g = subgraphs[i]
            choice = self.choice[i]

            bfs = nx.bfs_edges(g, choice)   # DFS would also work here
            for start, end in bfs:
                self.infections[end] = True
                states.append(self._infection_sort(self.infections.items()))
        states.append(self._infection_sort(self.infections.items()))
        return states

    def limited_infection(self, infect_max=10, infect_size=None):
        """
        We can look at this as virus propagation. As similar as this is with the total infection problem, we actually
        want to use a completely different approach

        We want to start the infection at the most central node.
            * The core idea here is that since we want to limit our network so that it only affects "groups" of people,
            we want to initially start at the most connected person so that way there's less of a chance of only halfway
            infecting a network.
            * THIS IS MUCH BETTER THAN RANDOM CHOICE (probably?)

        Decaying Markov Chain
            * Probabilities decay proportionally to centrality. In essence, the further away from the center you
            get, the less chance you have of being infected.
                * Using 1 / (x+c) where x is size of infection and c is centrality for the node
            * Combined with flat threshhold?
        """
        scores, node = self._graph_centrality()
        self.choice = [node]  # We want this central node to be choice
        self._infection_list()  # Need to refresh infection status

        states = [self._infection_sort(self.infections.items())]

        markovchain = self._get_markovchain()
        cnode = node
        while True:   # Rebalances weights /every/ cycle
            print(cnode, self.infections[cnode], self._infection_size())
            cnode = np.random.choice(np.arange(self.graph.shape[0]), p=markovchain[:, cnode])
            self.infections[cnode] = True
            size = self._infection_size()
            weights = [5 / (size + scores[i][1]) for i in range(self.graph.shape[0])]
            markovchain[:, cnode] *= weights
            markovchain[cnode, cnode] += 1 - markovchain[:, cnode].sum()
            states.append(self._infection_sort(self.infections.items()))
            if size >= infect_max:
                break

        states.append(self._infection_sort(self.infections.items()))

        return states

    def _get_markovchain(self):
        """
        Returns randomized initial markov chain for graph.

        How this works: in order to determine next position, randomly pick entry from column corresponding to current
        position.
        """
        # Markov chain is initially randomized probabilities
        markovchain = ((self.graph + np.eye(self.graph.shape[0])) * np.random.random(self.graph.shape))
        # Need to normalize (columns need to sum to 1)
        markovchain /= markovchain.sum(axis=0)
        return markovchain

    def _infection_size(self):
        count = 0
        for key, value in self.infections.items():
            if value is True:
                count += 1
        return count

    def _graph_centrality(self):
        """
        Finds the most central node in the graph.
        https://en.wikipedia.org/wiki/Centrality

        This uses the eigenvector centrality:
        https://networkx.github.io/documentation/latest/reference/generated/networkx.algorithms.centrality.eigenvector_centrality.html#networkx.algorithms.centrality.eigenvector_centrality
        """
        centrality_scores = [(a, b) for a, b in nx.eigenvector_centrality(self.nxgraph).items()]
        central_node = max(centrality_scores, key=lambda tup: tup[1])[0]  # We now use this as our choice
        return centrality_scores, central_node

    def naive_limited_infection(self):
        """
        NAIVE LIMITED INFECTION

        Just does a DFS with decay factor
        """
        states = [self._infection_sort(self.infections.items())]

        subgraphs = list(nx.weakly_connected_component_subgraphs(self.nxgraph))
        for i in range(len(subgraphs)):
            g = subgraphs[i]
            choice = self.choice[i]

            bfs = nx.bfs_edges(g, choice)   # DFS would also work here
            cnode = choice
            for start, end in bfs:
                cnode = start
                weight = len(nx.shortest_path(self.nxgraph, source=choice, target=cnode))
                if np.random.random() > np.exp(-weight + 1.5):
                    break
                self.infections[end] = True
                states.append(self._infection_sort(self.infections.items()))
        states.append(self._infection_sort(self.infections.items()))

        return states

    def animate_infection(self, states):
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        pos = nx.spring_layout(self.nxgraph)

        colors = np.zeros((len(states[0]), len(states)))
        for i in range(len(states)):
            colors[:, i] = [0 if infection is False else 1 for node, infection in states[i]]

        nodes = nx.draw_networkx_nodes(self.nxgraph, pos=pos, node_color=colors[:, 0])
        edges = nx.draw_networkx_edges(self.nxgraph, pos=pos)
        labels = nx.draw_networkx_labels(self.nxgraph, pos=pos, font_color='w')

        def animate(i):
            nodes = nx.draw_networkx_nodes(self.nxgraph, pos=pos, node_color=colors[:, i])
            return nodes, edges

        def init():
            return nodes, edges

        ani = animation.FuncAnimation(fig, animate, np.arange(len(states)), init_func=init,
                interval=50)

        if self.write:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Will Farmer'), bitrate=1800)
            ani.save('infection.mp4', writer=writer)

        plt.show()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--refresh', action='store_true', default=False,
                            help='Refresh Graph')
    parser.add_argument('-a', '--animate', action='store_true', default=False,
                            help='Animate Infection')
    parser.add_argument('-w', '--write', action='store_true', default=False,
                            help='Save Animation')
    parser.add_argument('-n', '--nodes', type=int, default=20,
                            help='How many nodes to generate')
    parser.add_argument('-p', '--prob', type=float, default=0.2,
                            help='Edge Probability')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    sys.exit(main())
