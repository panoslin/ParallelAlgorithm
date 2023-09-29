#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 9/28/23
# IDE: PyCharm
from collections import defaultdict
from functools import cache
from typing import List


class TSP:
    """
    Travelling Salesman Problem
    Solve using DP in
    TC: 2 ** n * n ** 2
    SC: 2 ** n * n
    """
    # number of nodes from the graph
    n = 0
    # adjacent matrix
    weights = None
    start_node = 0
    prev: List[List[int]] = []
    end_mask = 0

    def travel(self, weights: List[List[int]]):
        self.n = len(weights)
        self.weights = weights
        self.start_node = 0
        # a bit mask indicating all nodes have been visited
        self.end_mask = 1 << self.n
        self.prev = [[-1] * self.end_mask for _ in range(self.n)]
        return self.dp(1 << self.start_node, self.start_node)

    @cache
    def dp(self, taken, current_node) -> int:
        """
        minimal cost travelling from 0 node to all others nodes and back to 0 node
        given the travelled nodes as taken
        represented by a bit mask

        TC: n * 2 ** n ** 2

        return minimal cost as integer
        """
        if taken + 1 == self.end_mask:
            return self.weights[current_node][0]

        min_cost = float('inf')
        neighbor = -1
        for neighbor in range(self.n):
            # neighbor node is not visited
            if not taken & (1 << neighbor):
                # cost is lesser
                cost = self.weights[current_node][neighbor] + self.dp(taken | 1 << neighbor, neighbor)
                if cost < min_cost:
                    min_cost = cost

        self.prev[current_node][taken] = neighbor
        return min_cost


if __name__ == '__main__':

    for tc, x in [
        ([[0, 5, 8], [4, 0, 8], [4, 5, 0]], 17),
        ([[10000.0, 10000.0, 10000.0, 8.0, 10000.0, 10.0], [10000.0, 10000.0, 10000.0, 10000.0, 2.0, 12.0], [10000.0, 10000.0, 10000.0, 6.0, 4.0, 10000.0], [8.0, 10000.0, 6.0, 10000.0, 10000.0, 10000.0], [10000.0, 2.0, 4.0, 10000.0, 10000.0, 10000.0], [10.0, 12.0, 10000.0, 10000.0, 10000.0, 10000.0]], 42),
        ([[0, 6, 3, 4, 4], [4, 0, 4, 3, 7], [4, 3, 0, 4, 6], [2, 6, 4, 0, 6], [3, 5, 4, 4, 0]], 16),
        ([[0, 17, 15, 16, 16, 15, 19, 19, 16, 18, 20], [10, 0, 15, 16, 15, 12, 19, 19, 16, 18, 20],[10, 17, 0, 16, 16, 16, 19, 19, 16, 18, 20], [10, 17, 14, 0, 16, 16, 19, 19, 16, 3, 8],[10, 17, 15, 1, 0, 16, 19, 19, 16, 4, 9], [10, 17, 15, 16, 16, 0, 19, 19, 6, 18, 20],[10, 17, 15, 3, 2, 16, 0, 19, 15, 6, 11], [10, 11, 15, 16, 15, 16, 19, 0, 16, 18, 20],[8, 17, 15, 16, 16, 16, 19, 19, 0, 18, 20], [10, 17, 11, 16, 16, 16, 19, 19, 16, 0, 5],[10, 17, 6, 16, 16, 16, 19, 18, 16, 18, 0]],92)
    ]:
        tsp = TSP()
        y = tsp.travel(tc)
        print(y)
        assert x == y
