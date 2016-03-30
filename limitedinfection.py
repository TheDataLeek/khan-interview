#!/usr/bin/env python3.5


"""
Khan Academy's Limited Infection Problem
========================================
See ./README.md for details

Some notes:
    * Some directed edge from A->B indicates that A coaches B.
    * The fact that the network is directed actually isn't super important, as the mere fact that
    two nodes are connected is important to keep track of.
"""


import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx


# I have a big screen
np.set_printoptions(linewidth=160)


def main():
    """
    In the event this is run from the command line, this main function provides sample use.
    """
    args = get_args()

    infection = NetworkInfection(args.nodes, args.prob, args.write, refresh=args.refresh)
    infection.load()
    infection.choose()
    if args.limited:
        states = infection.limited_infection(args.size, args.stickiness)
    else:
        states = infection.total_infection()
    if args.animate:
        infection.animate_infection(states)


class NetworkInfection(object):
    """
    Network Infection

    Responsible for management of our network and current infection state.

    :param nodecount: int => Number of nodes in generated Network
    :param prob: float => Probability in generated network for edge to be created
    :param write: bool => Whether or not to save the network animation as .mp4

    :returns: <NetworkInfection>
    """
    def __init__(self, nodecount, prob, write,
                 filename='./testnetwork.npy',
                 refresh=False, choose_node=False):
        self.networkfile = filename
        self.graph       = None
        self.nxgraph     = None
        self.choice      = choose_node
        self.write       = write
        self.infections  = None
        self.subgraphs   = False

        if refresh:
            gen_new_random_graph(nodecount, prob)
            self.filename = './testnetwork.npy'

    def load(self):
        """
        Loads adjacency matrix network from provided .npy file.

        filename is set in class instance.
        """
        self.graph   = np.load(self.networkfile)
        self.nxgraph = nx.DiGraph(self.graph)
        if nx.number_weakly_connected_components(self.nxgraph) > 1:
            self.subgraphs = True

    def show(self):
        """
        Draws the current network using Matplotlib.

        NOTE: BLOCKING. If this is done, it will stop any code execution.
        """
        plt.figure()
        nx.draw(self.nxgraph, pos=nx.spring_layout(self.nxgraph))
        plt.show()


    def choose(self):
        """
        Selects a random node to initially infect for every independent subgraph.

        Then updates the list of infected nodes based on these choices.
        """
        if isinstance(self.choice, bool):   # Prevent from re-picking
            if self.choice:
                self.choice = [input('Select Node(s)')]   # Not really intended for use
            else:
                self.choice = []
                for graph in nx.weakly_connected_component_subgraphs(self.nxgraph):
                    self.choice.append(np.random.choice(graph.nodes()))
        self._infection_list()

    def _infection_list(self):
        """
        Updates infection list.
        """
        self.infections = {n:(True if n in self.choice else False) for n in self.nxgraph.nodes()}

    def total_infection(self):
        """
        This part is straightforward, just simple DFS graph traversal on each independent subgraph.
        """
        states = [dict_item_sort(self.infections.items())]

        subgraphs = list(nx.weakly_connected_component_subgraphs(self.nxgraph))
        for i, graph in enumerate(subgraphs):
            choice = self.choice[i]

            bfs = nx.bfs_edges(graph, choice)   # DFS would also work here
            for start, end in bfs:
                self.infections[end] = True
                states.append(dict_item_sort(self.infections.items()))
        states.append(dict_item_sort(self.infections.items()))
        return states

    def limited_infection(self, infection_size, stickiness):
        """
        We can look at this as virus propagation. As similar as this is with the total infection
        problem, we actually want to use a completely different approach

        We want to start the infection at the most central node.
            * The core idea here is that since we want to limit our network so that it only affects
            "groups" of people, we want to initially start at the most connected person so that way
            there's less of a chance of only halfway infecting a network.
            * THIS IS MUCH BETTER THAN RANDOM CHOICE (probably?)

        Decaying Markov Chain
            * Probabilities decay proportionally to centrality. In essence, the further away from
            the center you get, the less chance you have of being infected.
                * Using 1 / ((x+c)**2) where x is size of infection and c is centrality for the node
            * Combined with flat threshhold
            * Breakout condition is if it's bounced around inside the network 3 times
        """
        # If no infection size, set to max size of graph and rely on decay process
        if infection_size == -1:
            infection_size = len(self.nxgraph.nodes())

        scores, node = self._graph_centrality()
        self.choice = [node]  # We want this central node to be choice
        self._infection_list()  # Need to refresh infection status

        states = [dict_item_sort(self.infections.items())]

        markovchain = self._get_markovchain()
        cnode = node
        size = self._infection_size()
        network_stickiness = 0
        while size < infection_size:   # Rebalances cnode weights /every/ cycle
            # Choose next node to jump to
            pnode = cnode
            cnode = np.random.choice(np.arange(self.graph.shape[0]), p=markovchain[:, cnode])
            # Check Stickiness
            if self.infections[cnode] is False:
                network_stickiness = 0
            else:
                network_stickiness += 1
            if network_stickiness >= stickiness:
                break
            # Set its status to "infected"
            self.infections[cnode] = True
            # Rebalance current choices.
            # As size of infected network increases, and as we get further away from the center
            # lower probs Increase probability of a backjump (to stay close to center and keep
            # infecting from there)
            size = self._infection_size()
            weights = np.array([1 / ((size + scores[i][1])**2)
                                for i in range(self.graph.shape[0])])
            weights /= weights.sum()
            markovchain[:, cnode] = weights
            markovchain[pnode, cnode] += 1 - markovchain[:, cnode].sum()

            states.append(dict_item_sort(self.infections.items()))

        states.append(dict_item_sort(self.infections.items()))

        print('Final Infection Size: {}'.format(self._infection_size()))

        return states

    def _get_markovchain(self):
        """
        Returns randomized initial markov chain for graph.

        How this works: in order to determine next position, randomly pick entry from column
        corresponding to current position.
        """
        # Markov chain is initially randomized probabilities
        markovchain = self.graph + np.eye(self.graph.shape[0])
        markovchain *= np.random.random(self.graph.shape)
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
        # We now use this central node as our choice
        central_node = max(centrality_scores, key=lambda tup: tup[1])[0]
        return centrality_scores, central_node

    def naive_limited_infection(self):
        """
        NAIVE LIMITED INFECTION

        Just does a DFS with decay factor
        """
        states = [dict_item_sort(self.infections.items())]

        subgraphs = list(nx.weakly_connected_component_subgraphs(self.nxgraph))
        for i, graph in enumerate(subgraphs):
            choice = self.choice[i]

            bfs = nx.bfs_edges(graph, choice)   # DFS would also work here
            cnode = choice
            for start, end in bfs:
                cnode = start
                weight = len(nx.shortest_path(self.nxgraph, source=choice, target=cnode))
                if np.random.random() > np.exp(-weight + 1.5):
                    break
                self.infections[end] = True
                states.append(dict_item_sort(self.infections.items()))
        states.append(dict_item_sort(self.infections.items()))

        return states

    def animate_infection(self, states):
        """
        Animate Infection Spread

        :param states: list => 2D list of network states
        """
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
            """ Change plot state for next frame """
            nodes = nx.draw_networkx_nodes(self.nxgraph, pos=pos, node_color=colors[:, i])
            return nodes, edges

        def init():
            """ First animation Frame """
            return nodes, edges

        ani = animation.FuncAnimation(fig, animate, np.arange(len(states)), init_func=init,
                                      interval=50)

        if self.write:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Will Farmer'), bitrate=1800)
            ani.save('infection.mp4', writer=writer)

        plt.show()


def gen_new_random_graph(nodecount, prob):
    """
    Generate a new random graph using binomial generation.

    Will save new network to file.
    """
    newgraph = nx.binomial_graph(nodecount, prob)
    np.save('testnetwork.npy', nx.adjacency_matrix(newgraph).todense())


def dict_item_sort(dlist):
    """
    Provides sorted version of infection list. Need to sort list form of infection as
    dictionaries are unsorted
    """
    return sorted(dlist, key=lambda tup: tup[0])


def get_args():
    """
    Get command line arguments with argparse

    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--refresh', action='store_true', default=False,
                        help='Refresh Graph')
    parser.add_argument('-a', '--animate', action='store_true', default=False,
                        help='Animate Infection')
    parser.add_argument('-w', '--write', action='store_true', default=False,
                        help='Save Animation')
    parser.add_argument('-l', '--limited', action='store_true', default=False,
                        help='Limited Infection or Total? -l indicates limited')
    parser.add_argument('-n', '--nodes', type=int, default=20,
                        help='How many nodes to generate')
    parser.add_argument('-p', '--prob', type=float, default=0.2,
                        help='Edge Probability')
    parser.add_argument('-s', '--size', type=int, default=-1,
                        help='How many nodes to infect')
    parser.add_argument('-k', '--stickiness', type=int, default=3,
                        help='How sticky the Markov Process is')
    args = parser.parse_args()
    if args.size != -1:
        args.stickiness = 100
    return args


if __name__ == '__main__':
    sys.exit(main())
