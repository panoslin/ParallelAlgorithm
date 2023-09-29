#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 9/28/23
# IDE: PyCharm
import concurrent.futures
from collections import defaultdict
from functools import cache
from typing import List, Dict, Tuple
import os
import math


class TSPThreading:
    """
    Reference from https://github.com/fillipe-gsm/python-tsp/blob/master/python_tsp/exact/dynamic_programming.py
    Travelling Salesman Problem
    Solve by DP in
    TC: 2 ** n * n ** 2
    SC: 2 ** n * n
    """

    def __init__(self, weights: List[List[int]]):
        # adjacent matrix
        self.weights = weights
        self.path: Dict[Tuple, int] = {}
        self.thread_count = math.factorial(len(weights))
        self.executor: concurrent.futures.Executor = None

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            self.executor = executor
            return self.travel()

    def travel(self):
        taken = frozenset(range(1, len(self.weights)))

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
    def dist(self, current_node: int, remaining_nodes: frozenset) -> float:
        if not remaining_nodes:
            return self.weights[current_node][0]

        # Store the costs in the form (neighbor, dist(neighbor, taken))
        costs = []
        # Start the load operations and mark each future with its parameters
        future_to_idx = {
            self.executor.submit(
                self.dist,
                current_node=neighbor,
                remaining_nodes=remaining_nodes.difference({neighbor})
            ): (neighbor, self.weights[current_node][neighbor])
            for neighbor in remaining_nodes
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            neighbor, weight = future_to_idx[future]
            try:
                costs.append((neighbor, weight + future.result()))
            except Exception as exc:
                print(f'{neighbor}, {weight} generated an exception: {exc}')

        optimal_neighbor, min_cost = min(costs, key=lambda x: x[1])
        self.path[(current_node, remaining_nodes)] = optimal_neighbor

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
          [10, 17, 6, 16, 16, 16, 19, 18, 16, 18, 0]], 92),
        ([
             [0, 1, 2, 3, 4, 5, 6, 7, 8],
             [1, 0, 9, 10, 11, 12, 13, 14, 15],
             [2, 9, 0, 16, 17, 18, 19, 20, 21],
             [3, 10, 16, 0, 22, 23, 24, 25, 26],
             [4, 11, 17, 22, 0, 27, 28, 29, 30],
             [5, 12, 18, 23, 27, 0, 31, 32, 33],
             [6, 13, 19, 24, 28, 31, 0, 34, 35],
             [7, 14, 20, 25, 29, 32, 34, 0, 36],
             [8, 15, 21, 26, 30, 33, 35, 36, 0]
         ], 154),
        ([
             [0, 1, 2, 3, 4, 5, 6, 7],
             [1, 0, 8, 9, 10, 11, 12, 13],
             [2, 8, 0, 14, 15, 16, 17, 18],
             [3, 9, 14, 0, 19, 20, 21, 22],
             [4, 10, 15, 19, 0, 23, 24, 25],
             [5, 11, 16, 20, 23, 0, 26, 27],
             [6, 12, 17, 21, 24, 26, 0, 28],
             [7, 13, 18, 22, 25, 27, 28, 0]
         ], 108),
    ]:
        tsp = TSPThreading(tc)
        y = tsp.run()
        print(y)
        # print(tsp.get_path())

        assert expected == y[1]
