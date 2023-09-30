#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 9/28/23
# IDE: PyCharm
"""
mpiexec -np 4 python tsp/tsp_mpi.py
"""
import concurrent.futures
import os
from itertools import permutations
from typing import List, Tuple
from mpi4py import MPI


class TSP:
    """
    Travelling Salesman Problem
    Solve by DP in
    TC: 2 ** n * n ** 2
    SC: 2 ** n * n
    """

    def __init__(self, weights: List[List[int]]):
        self.weights = weights
        self.thread_count = os.cpu_count()

    def travel(self):

        # comm = MPI.COMM_WORLD
        # rank = comm.Get_rank()
        # size = comm.Get_size()
        #
        # if rank == 0:
        #     data = range(10)
        #     comm.send(data, dest=1, tag=11)
        #     print("process {} send {}...".format(rank, data))
        # else:
        #     data = comm.recv(source=0, tag=11)
        #     print("process {} recv {}...".format(rank, data))

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

        for mask in range(1 << n):
            self.helper(n=n, mask=mask, dp=dp)

        return dp[-1][0]

    def helper(self, n, mask: int, dp) -> List[int]:
        """
        each processor will need the dp of which have bit distance of 1 comparing to mask

        """
        result: List[int] = [float('inf')] * n
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
                    dp[-1][i][0] + self.weights[i][0],
                    dp[-1][i][1]
                )
        return result


if __name__ == '__main__':
    from testcases import testcases

    for tc, expected in testcases:
        tsp = TSP(tc)
        y = tsp.travel()
        print(y)

        assert expected == y[0]
