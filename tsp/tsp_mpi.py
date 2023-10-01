#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 9/28/23
# IDE: PyCharm
"""
mpiexec -n 4 python -m mpi4py.futures tsp/tsp_mpi2.py
"""
from concurrent.futures import Future
from functools import partial
from itertools import permutations
from typing import List, Tuple

from mpi4py import MPI, futures


def travel(n, mask: int, dp: List[List[Tuple[float, List]]], weights: List[List[int]]) -> List[int]:
    """
    each processor will need the dp of which have bit distance of 1 comparing to mask

    """

    result: list = list(dp[mask])

    nodes_to_be_visited = [j for j in range(n) if mask & (1 << j)]
    for dest, src in permutations(nodes_to_be_visited, 2):
        state_dest_not_visited = mask ^ (1 << dest)
        result[dest] = min(
            result[dest],
            (
                dp[state_dest_not_visited][src][0] + weights[src][dest],
                dp[state_dest_not_visited][src][1] + [dest]
            ),
            key=lambda x: x[0]
        )

    return result


def callback(req: Future, mask):
    dp[mask] = req.result()


if __name__ == '__main__':
    from testcases import testcases
    import time

    # init MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    with futures.MPIPoolExecutor() as executor:
        for weights, expected in testcases:
            start_time = time.time()

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

            reqs = []
            count = 2
            for mask in range(1, 1 << n):
                req: Future = executor.submit(
                    travel,
                    n=n,
                    mask=mask,
                    dp=dp,
                    weights=weights
                )
                req.add_done_callback(partial(callback, mask=mask))
                reqs.append(req)
                count -= 1
                if count <= 0:
                    futures.wait(reqs)
                    reqs = []
                    count = 2

            futures.wait(reqs)

            print(
                f'\nFinish TSP with result {dp[-1][0]}\n'
                f'time taken to process {len(weights)} nodes TSP '
                f'with {size} processor: '
                f'{time.time() - start_time}\n'
                f'*****************************************************************************************************'
            )
        assert expected == dp[-1][0][0]
