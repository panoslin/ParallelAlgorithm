#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 9/28/23
# IDE: PyCharm
from typing import List, Tuple
from itertools import permutations


class TSP:
    """
    Reference from https://leetcode.com/discuss/general-discussion/1125779
    Travelling Salesman Problem
    Solve by DP in
    TC: 2 ** n * n ** 2
    SC: 2 ** n * n
    """

    @staticmethod
    def travel(weights: List[List[int]]):
        start_node = 0
        n = len(weights)
        # dp[state][node]
        # minimal cost/path given state and previous visited node
        dp: List[List[Tuple[float, List]]] = [
            [(float('inf'), [])] * n
            for _ in range(1 << n)
        ]
        for i in range(n):
            # first node taken
            # should be after a start node
            dp[1 << i][i] = (weights[start_node][i], [i])

        for mask in range(1 << n):
            nodes_to_be_visited = [j for j in range(n) if mask & (1 << j)]
            for dest, src in permutations(nodes_to_be_visited, 2):
                state_dest_not_visited = mask ^ (1 << dest)
                dp[mask][dest] = min(
                    dp[mask][dest],
                    (
                        dp[state_dest_not_visited][src][0] + weights[src][dest],
                        dp[state_dest_not_visited][src][1] + [dest]
                    ),
                    key=lambda x: x[0]
                )

            # reach to the last node
            if mask + 1 == 1 << n:
                for i in range(n):
                    # go back to start node
                    dp[-1][i] = (
                        dp[-1][i][0] + weights[i][start_node],
                        dp[-1][i][1]
                    )

        return dp[-1][0]


if __name__ == '__main__':
    from testcases import testcases

    for tc, expected in testcases:
        tsp = TSP()
        y = tsp.travel(tc)
        print(y)

        assert expected == y[0]
