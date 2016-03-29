#!/usr/bin/env python3.5

import limitedinfection
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pytest

class TestInfection(object):
    @pytest.fixture()
    def infection(self):
        return limitedinfection.NetworkInfection('./testnetwork.npy')

    def test_init(self, infection):
        assert(infection.networkfile == './testnetwork.npy')

    def test_load(self, infection):
        infection.load()
        assert(infection.graph is not None)
        assert(infection.nxgraph is not None)

    def test_choice(self, infection):
        infection.load()
        assert(infection.choice is False)
        infection.choose()
        oldchoice = infection.choice
        assert(infection.choice in infection.nxgraph.nodes())
        infection.choose()
        assert(oldchoice == infection.choice)
