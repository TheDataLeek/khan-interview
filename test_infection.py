#!/usr/bin/env python3.5

import limitedinfection
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import functools
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
        for c in infection.choice:
            assert(c in infection.nxgraph.nodes())
        infection.choose()
        assert(oldchoice == infection.choice)

    def test_initial_infection(self, infection):
        infection.load()
        infection.choose()
        for node, status in infection.infections.items():
            if node in infection.choice:
                assert(status is True)
            else:
                assert(status is False)

    def test_total_infection(self, infection):
        infection.load()
        infection.choose()
        states = infection.total_infection()
        # Assert everything infected at end
        assert(functools.reduce(lambda acc, x: acc and x,
                [status for node, status in states[-1]]))
