# Khan Academy Virus Propagation

[![Build Status](https://travis-ci.org/willzfarmer/khan-interview.svg?branch=master)](https://travis-ci.org/willzfarmer/khan-interview)

Part of the interview process at Khan Academy is to complete their interview
project. This project (in a nutshell) deals with virus propagation through a
directed network. I will be using the terms "graph" and "network"
interchangeably, [but I'm referring to the same thing for
both](https://en.wikipedia.org/wiki/Graph_theory).

*We will be using [`pytest`](http://pytest.org/latest/),
[`numpy`](http://www.numpy.org/), [`matplotlib`](http://matplotlib.org/), and
[`networkx`](https://networkx.github.io/) as external libraries for this
analysis. I will be using the following common abreviations:*

```python
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
```

## Setup

Right off the bat we need to define our network. For adaptability we will assume
that a local
[`.npy`](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.load.html)
file exists and contains an [adjacency
matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) that defines the nodes
and connections in our directed graph. We don't keep track of any unique names
for the nodes, and instead just refer to them as numbers for simplicity.

## Choosing an Initial Infected Node

The first part of this is to figure out where exactly to kick off our infection.
There are a couple ways to do this, and I use two main ones.

### The Naive Approach

The easiest way is to just pick a node at random. Using [`numpy`'s choice
function](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.random.choice.html)
and [`networkx`'s `nodes()`
function](https://networkx.github.io/documentation/latest/reference/generated/networkx.Graph.nodes.html)
we can easily select a single starting node.

```python
np.random.choice(graph.nodes())
```

The one possible issue with this approach is that if we have [independent
subgraphs](https://en.wikipedia.org/wiki/Connectivity_%28graph_theory%29) we
need to examine each separately, and pick a random node for every subgraph.

```python
self.choice = []
for graph in nx.weakly_connected_component_subgraphs(self.nxgraph):
    self.choice.append(np.random.choice(graph.nodes()))
```

### The More Complex Approach

In graph theory there's a concept of
[centrality](https://en.wikipedia.org/wiki/Centrality). In essence, centrality
is the concept of how "important" each node in the graph is. There are a ton of
different ways to define this concept of "importance", from the number of edges
of each node, to examining the [leading
eigenvalue](https://en.wikipedia.org/wiki/Centrality#Eigenvector_centrality).
Using this approach we can find the "most important" node and infect it first.

```python
centrality_scores = [(a, b) for a, b in nx.eigenvector_centrality(self.nxgraph).items()]
central_node = max(centrality_scores, key=lambda tup: tup[1])[0]
```

The reason behind focusing on centrality for the initial node choice is that in
the type of network that we have, we want to infect teachers and every student
first. By selecting a more central node as our starting point, we can hopefully
avoid halfway infecting [strongly
connected](https://en.wikipedia.org/wiki/Strongly_connected_component) parts of
the network.

Again, we need to pick one of these per subgraph, so that way we aren't limited
by unconnected portions of the graph.

## Part A - Total Infections

> Ideally we would like every user in any given classroom to be using the same
> version of the site.  Enter “infections”. We can use the heuristic that each
> teacher­student pair should be on the same version of the site. So if A
> coaches B and we want to give A a new feature, then B should also get the new
> feature. Note that infections are transitive ­ if B coaches C, then C should
> get the new feature as well.  Also, infections are transferred by both the
> “coaches” and “is coached by” relations.  Now implement the infection
> algorithm. Starting from any given user, the entire connected component of the
> coaching graph containing that user should become infected.

The solution to this part is very straightforward. We need to first select a
node as our initial infected node, and then just do a [Breadth First
Search](https://en.wikipedia.org/wiki/Breadth-first_search) from that node (a
[Depth First Search](https://en.wikipedia.org/wiki/Depth-first_search) would
also work). As we travel through the nodes we "infect" it until every node has
the virus.

![Total Infection](https://raw.githubusercontent.com/willzfarmer/khan-interview/master/animations/totalinfection.gif)

The code for this is also fairly straightforward, especially as `networkx` has
graph traversal algorithms built in.

```python
subgraphs = list(nx.weakly_connected_component_subgraphs(self.nxgraph))
for i, graph in enumerate(subgraphs):
    choice = self.choice[i]
    bfs = nx.bfs_edges(graph, choice)
    for start, end in bfs:
        self.infections[end] = True
```

## Part B - Limited Infection

> We would like to be able to infect close to a given number of users. Ideally
> we’d like a coach and all of their students to either have a feature or not.
> However, that might not always be possible. Implement a procedure for limited
> infection. You will not be penalized for interpreting the specification as you
> see fit. There are many design choices and tradeoffs, so be prepared to
> justify your decisions.

This is a little harder, and there are many different ways to approach the
problem.

![Limited Infection](https://raw.githubusercontent.com/willzfarmer/khan-interview/master/animations/limitedinfections.gif)
