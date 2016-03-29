#!/usr/bin/env python3.5

import limitedinfection
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pytest

class TestInfection(object):
    @pytest.fixture
    def network():
        newgraph = nx.binomial_graph(10, 0.1, directed=True)
        nx.draw(newgraph)
