# Topology-Apps

## Table_of_contents
- [Graph_Topology](#graph_topology)
  - [Table of content](#table_of_contents)
  - [Overview](#overview)
  - [Requirement](#requirement)
  - [Getting started](#getting_started)
  - [Features](#features)
  - [Roadmap](#roadmap)
  - [References](#references)

## Overview

This apps aims to visualise an input graph network file and label nodes representing different topological features of input graph.

Currently this apps is still in development and planed to include more features in the future.

## Requirement
This apps used the following package with a specific version.
- python 3.9.16
- sys
- pandas 1.5.3
- networkx 3.0
- PuLp 2.7.0
- PyQt5 12.11.0
- matplotlib 3.6.1

## Getting_started
2 example network edge files are provided in this repository, see [dataset]

Format of network edge files is a tab separated text file, header of files are [regulator, target].

## Features
- File menu
  - Open graph
  - Close graph: Clear current graph
- Edit menu
  - Remove self-loop: to remove all edges that having identical regulator and targets
  - Reverse graph: Reverse direction of all edges (Regulator becomes target, and vice versa)
- Compute menu
  - MDS: Compute Minimum Dominating Set (MDS) of the graph using integer linear programming (ILP) approach
  - MCDS: Compute Minimum Connected Dominating Set (MCDS) of the largest strongly connected component (LSCC) of the graph using Heuristic approach
  - out-degree Hubs: Compute the top 10% of nodes having the highest out-degree (outgoing edges)
  - in-degree Hubs: Compute the top 10% of nodes having the highest in-degree (incoming edges)
  - LCC: Label all nodes present in the largest connected component (LCC) of graph

## Roadmap
- [ ] Implement option menu to adjust appearance of nodes, edges, and labeled nodes
- [ ] Include more topological features
  - [ ] Label out- and in- degree of each node
  - [ ] Network certainities
  - [ ] DAG (directed acyclic graph) components
  - [ ] Cliques
  - [ ] Strong and weak communities
  - [ ] Minimum cut edges
- [ ] Re-implement an interactive representation of graphs
  - [ ] Remove specific nodes
  - [ ] Move position of nodes
  - [ ] Add new edges between nodes

## References
[1] Nazarieh, M., Wiese, A., Will, T., Hamed, M., & Helms, V. (2016). Identification of key player genes in gene regulatory networks. BMC Systems Biology, 10, 1-12.
