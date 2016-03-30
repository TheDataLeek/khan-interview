# Khan Academy Virus Propagation

[![Build Status](https://travis-ci.org/willzfarmer/khan-interview.svg?branch=master)](https://travis-ci.org/willzfarmer/khan-interview)

Part of the interview process at Khan Academy is to complete their interview
project. This project (in a nutshell) deals with virus propagation through a
directed network.

*We will be using [`pytest`](http://pytest.org/latest/),
[`numpy`](http://www.numpy.org/), [`matplotlib`](http://matplotlib.org/), and
[`networkx`](https://networkx.github.io/) as external libraries for this
analysis.*

## Setup

Right off the bat we need to define our network. For adaptability we will assume
that a local
[`.npy`](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.load.html)
file exists and contains an [adjacency
matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) that defines the nodes
and connections in our directed graph. We don't keep track of any unique names
for the nodes, and instead just refer to them as numbers for simplicity.

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

## Part B - Limited Infection

> We would like to be able to infect close to a given number of users. Ideally
> we’d like a coach and all of their students to either have a feature or not.
> However, that might not always be possible. Implement a procedure for limited
> infection. You will not be penalized for interpreting the specification as you
> see fit. There are many design choices and tradeoffs, so be prepared to
> justify your decisions.

![Limited Infection](https://raw.githubusercontent.com/willzfarmer/khan-interview/master/animations/limitedinfections.gif)
