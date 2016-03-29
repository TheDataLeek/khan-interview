#!/usr/bin/env python3.5


import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def main():
    newgraph = nx.binomial_graph(10, 0.1, directed=True)
    nx.draw(newgraph)


def foo():
    return 5




if __name__ == '__main__':
    sys.exit(main())
