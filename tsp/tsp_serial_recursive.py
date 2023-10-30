#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 9/28/23
# IDE: PyCharm
from collections import defaultdict
from functools import cache
from typing import List, Dict, Tuple


class TSP:
    """
    Reference from https://github.com/fillipe-gsm/python-tsp/blob/master/python_tsp/exact/dynamic_programming.py
    Travelling Salesman Problem
    Solve by DP in
    TC: 2 ** n * n ** 2
    SC: 2 ** n * n
    """
    # adjacent matrix
    weights = None
    path: Dict[Tuple, int] = {}

    def travel(self, weights: List[List[int]]):
        """
        Calculate the minimal cost/path for traveling through all nodes and back to the start node

        Args:
            weights (List[List[int]]): A 2D list representing the weights between nodes.

        Returns:
            Tuple[float, List]: A tuple containing the minimal cost and the path.
        """
        remaining_nodes = frozenset(range(1, len(weights)))
        self.weights = weights

        # Step 1: get minimum distance
        best_distance = self.dist(0, remaining_nodes)

        # Step 2: get path with the minimum distance
        current_node = 0  # start at the origin
        best_path = [0]
        while remaining_nodes:
            current_node = self.path[(current_node, remaining_nodes)]
            best_path.append(current_node)
            remaining_nodes = remaining_nodes.difference({current_node})

        return best_distance, best_path

    @cache
    def dist(self, current_node: int, remaining_nodes: frozenset) -> float:
        """
        return min cost from current_node traversing all remaining nodes
        :param current_node:
        :param remaining_nodes:
        :return:
        """
        if not remaining_nodes:
            return self.weights[current_node][0]

        # Store the costs in the form (neighbor, dist(neighbor, remaining_nodes))
        costs = [
            (
                neighbor,
                self.weights[current_node][neighbor] + self.dist(neighbor, remaining_nodes.difference({neighbor}))
            )
            for neighbor in remaining_nodes
        ]
        neighbor_with_min_cost, min_cost = min(costs, key=lambda x: x[1])
        self.path[(current_node, remaining_nodes)] = neighbor_with_min_cost

        return min_cost


if __name__ == '__main__':
    from testcases import testcases
    for tc, expected in testcases:
        tsp = TSP()
        y = tsp.travel(tc)
        print(y)
        # print(tsp.get_path())

        assert expected == y[0]
