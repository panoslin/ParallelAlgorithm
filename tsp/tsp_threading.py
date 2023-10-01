#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 9/28/23
# IDE: PyCharm
import concurrent.futures
import os
from itertools import permutations
from typing import List, Tuple


class TSP:
    """
    Reference from https://leetcode.com/discuss/general-discussion/1125779
    Travelling Salesman Problem
    Solve by DP in
    TC: 2 ** n * n ** 2
    SC: 2 ** n * n
    """

    def __init__(self, weights: List[List[int]]):
        self.weights = weights
        self.thread_count = min(os.cpu_count(), len(weights))

    def travel(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            start_node = 0
            n = len(self.weights)
            # dp[state][node]
            # minimal cost/path given state and previous visited node
            dp: List[List[Tuple[float, List]]] = [
                [(float('inf'), [])] * n
                for _ in range(1 << n)
            ]
            for i in range(n):
                # first node taken
                # should be after a start node
                dp[1 << i][i] = (self.weights[start_node][i], [i])

            # Start the load operations and mark each future with its parameters
            future_to_idx = {
                executor.submit(
                    self.helper,
                    n=n,
                    mask=mask,
                    dp=dp,
                    start_node=start_node
                ): mask
                for mask in range(1 << n)
            }
            concurrent.futures.wait(future_to_idx)

            return dp[-1][0]

    def helper(self, n, mask, dp, start_node):
        nodes_to_be_visited = [j for j in range(n) if mask & (1 << j)]
        for dest, src in permutations(nodes_to_be_visited, 2):
            state_dest_not_visited = mask ^ (1 << dest)
            dp[mask][dest] = min(
                dp[mask][dest],
                (
                    dp[state_dest_not_visited][src][0] + self.weights[src][dest],
                    dp[state_dest_not_visited][src][1] + [dest]
                ),
                key=lambda x: x[0]
            )

        # reach to the last node
        if mask + 1 == 1 << n:
            for i in range(n):
                # go back to start node
                dp[mask][i] = (
                    dp[-1][i][0] + self.weights[i][start_node],
                    dp[-1][i][1]
                )


if __name__ == '__main__':
    from testcases import testcases
    import time

    for tc, expected in testcases:
        start_time = time.time()
        tsp = TSP(tc)
        y = tsp.travel()
        print(
            f'\nFinish TSP with result {y}\n'
            f'time taken to process {len(tc)} nodes TSP '
            f'with {tsp.thread_count} threads: '
            f'{time.time() - start_time}\n'
            f'********************************************************************************************************'
        )

        assert expected == y[0]
