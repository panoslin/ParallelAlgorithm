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
        taken = frozenset(range(1, len(weights)))
        self.weights = weights

        # Step 1: get minimum distance
        best_distance = self.dist(0, taken)

        # Step 2: get path with the minimum distance
        current_node = 0  # start at the origin
        best_path = [0]
        while taken:
            current_node = self.path[(current_node, taken)]
            best_path.append(current_node)
            taken = taken.difference({current_node})

        return best_distance, best_path

    @cache
    def dist(self, current_node: int, taken: frozenset) -> float:
        if not taken:
            return self.weights[current_node][0]

        # Store the costs in the form (neighbor, dist(neighbor, taken))
        costs = [
            (neighbor, self.weights[current_node][neighbor] + self.dist(neighbor, taken.difference({neighbor})))
            for neighbor in taken
        ]
        optimal_neighbor, min_cost = min(costs, key=lambda x: x[1])
        self.path[(current_node, taken)] = optimal_neighbor

        return min_cost


if __name__ == '__main__':
    from testcases import testcases
    for tc, expected in testcases:
        tsp = TSP()
        y = tsp.travel(tc)
        print(y)
        # print(tsp.get_path())

        assert expected == y[0]
