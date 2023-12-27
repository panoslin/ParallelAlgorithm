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
        """
        Calculate the minimal cost/path for traveling through all nodes and back to the start node

        Args:
            weights (List[List[int]]): A 2D list representing the weights between nodes.

        Returns:
            Tuple[float, List]: A tuple containing the minimal cost and the path.
        """
        start_node = 0
        n = len(weights)

        # dp[state][node] representing the minimal cost/path,
        # visiting all nodes from bit mask `state` and last visited node is `node`
        dp: List[List[Tuple[float: List[int]]]] = [
            [(float('inf'), [])] * n
            for _ in range(1 << n)
        ]
        for i in range(n):
            # first node taken
            # should be after the start node
            state_only_i_visited = 1 << i
            dp[state_only_i_visited][i] = (weights[start_node][i], [i])

        for current_mask in range(1 << n):
            nodes_to_be_visited = [j for j in range(n) if current_mask & (1 << j)]

            # compare all possible permutation of nodes
            for dest, src in permutations(nodes_to_be_visited, 2):
                mask_dest_not_visited = current_mask ^ (1 << dest)
                dp[current_mask][dest] = min(
                    dp[current_mask][dest],
                    (
                        dp[mask_dest_not_visited][src][0] + weights[src][dest],
                        dp[mask_dest_not_visited][src][1] + [dest]
                    ),
                    key=lambda x: x[0]
                )

        # all nodes visited and back to the start node
        return dp[(1 << n) - 1][0]


if __name__ == '__main__':
    from testcases import testcases
    import time

    for tc, expected in testcases:
        tsp = TSP()
        start_time = time.time()
        y = tsp.travel(tc)

        print(
            f'\nFinish TSP with result {y}\n'
            f'time taken to process {len(tc)} nodes TSP '
            f'{time.time() - start_time}\n'
            f'********************************************************************************************************'
        )

        assert expected == y[0]
