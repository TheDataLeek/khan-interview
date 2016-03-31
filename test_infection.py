#!/usr/bin/env python3.5

import infection
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import functools
import pytest

class TestInfection(object):
    @pytest.fixture(scope='module')
    def infection(self):
        return limitedinfection.NetworkInfection(50, 0.08, './testnetwork.npy')

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

    def test_markovchain(self, infection):
        infection.load()
        chain = infection._get_markovchain()
        assert(chain.sum(axis=1).all() == 1)
