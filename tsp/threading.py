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

        return best_path, best_distance

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

    for tc, expected in [
        ([[0, 5, 8], [4, 0, 8], [4, 5, 0]], 17),
        ([[10000.0, 10000.0, 10000.0, 8.0, 10000.0, 10.0], [10000.0, 10000.0, 10000.0, 10000.0, 2.0, 12.0],
          [10000.0, 10000.0, 10000.0, 6.0, 4.0, 10000.0], [8.0, 10000.0, 6.0, 10000.0, 10000.0, 10000.0],
          [10000.0, 2.0, 4.0, 10000.0, 10000.0, 10000.0], [10.0, 12.0, 10000.0, 10000.0, 10000.0, 10000.0]], 42),
        ([[0, 6, 3, 4, 4], [4, 0, 4, 3, 7], [4, 3, 0, 4, 6], [2, 6, 4, 0, 6], [3, 5, 4, 4, 0]], 16),
        ([[0, 17, 15, 16, 16, 15, 19, 19, 16, 18, 20], [10, 0, 15, 16, 15, 12, 19, 19, 16, 18, 20],
          [10, 17, 0, 16, 16, 16, 19, 19, 16, 18, 20], [10, 17, 14, 0, 16, 16, 19, 19, 16, 3, 8],
          [10, 17, 15, 1, 0, 16, 19, 19, 16, 4, 9], [10, 17, 15, 16, 16, 0, 19, 19, 6, 18, 20],
          [10, 17, 15, 3, 2, 16, 0, 19, 15, 6, 11], [10, 11, 15, 16, 15, 16, 19, 0, 16, 18, 20],
          [8, 17, 15, 16, 16, 16, 19, 19, 0, 18, 20], [10, 17, 11, 16, 16, 16, 19, 19, 16, 0, 5],
          [10, 17, 6, 16, 16, 16, 19, 18, 16, 18, 0]], 92)
    ]:
        tsp = TSP()
        y = tsp.travel(tc)
        print(y)
        # print(tsp.get_path())

        assert expected == y[1]
