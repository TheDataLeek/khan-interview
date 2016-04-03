#!/usr/bin/env python3.5

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import functools
import pytest

import infection

class TestInfection(object):
    @pytest.fixture(scope='module')
    def infection(self):
        return infection.NetworkInfection(50, 0.08, './test/testnetwork.npy')

    def test_init(self, infection):
        assert infection.networkfile == './test/testnetwork.npy'

    def test_load(self, infection):
        infection.load()
        assert infection.graph is not None
        assert infection.nxgraph is not None

    def test_choose(self, infection):
        infection.load()
        assert infection.choice is False
        infection.choose()
        oldchoice = infection.choice
        for c in infection.choice:
            assert c in infection.nxgraph.nodes()
        infection.choose()
        assert oldchoice == infection.choice

    def test_infection_list(self, infection):
        infection.load()
        infection.choose()
        for node, status in infection.infections.items():
            if node in infection.choice:
                assert status is True
            else:
                assert status is False

    def test_total_infection(self, infection):
        infection.load()
        infection.choose()
        states = infection.total_infection()
        # Assert everything infected at end
        assert functools.reduce(lambda acc, x: acc and x,
                [status for node, status in states[-1]])

    def test_limited_infection(self, infection):
        infection.load()
        infection.choose()
        states = infection.limited_infection(10, 3)
        assert infection._infection_size() == 10

    def test_markovchain(self, infection):
        infection.load()
        chain = infection._get_markovchain()
        assert chain.sum(axis=1).all() == 1

    def test_infection_size(self, infection):
        infection.load()
        infection.choose()
        infection.total_infection()
        assert infection._infection_size() == len(infection.nxgraph.nodes())

    def test_naive_limited_infection(self, infection):
        infection.load()
        infection.choose()
        presize = infection._infection_size()
        infection.naive_limited_infection()
        assert presize != infection._infection_size()

def test_dict_item_sort():
    testlist = [(1, 5), (5, 3), (0, 2), (3, 3)]
    res = infection.dict_item_sort(testlist)
    assert res == [(0, 2), (1, 5), (3, 3), (5, 3)]
