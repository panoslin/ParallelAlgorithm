#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 9/28/23
# IDE: PyCharm
"""
mpiexec -n 4 python -m mpi4py.futures tsp/tsp_mpi.py
"""
import math
from itertools import permutations
from typing import List, Tuple
from mpi4py import MPI, futures
from functools import partial
from concurrent.futures import Future


def travel(
        current_mask: int,
        dp: List[List[Tuple[float, List]]],
        weights: List[List[int]]
) -> List[int]:
    """
    each processor will need the dp of which have hamming distance of 1 comparing to current_mask
    """
    n = len(dp)
    nodes_to_be_visited = [j for j in range(n) if current_mask & (1 << j)]
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

    return dp[current_mask]


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

            # dp[state][node] representing the minimal cost/path,
            # visiting all nodes from bit mask `state` and last visited node is `node`
            dp: List[List[Tuple[float, List]]] = [
                [(float('inf'), [])] * n
                for _ in range(1 << n)
            ]
            for i in range(n):
                # first node taken
                # should be after the start node
                state_only_i_visited = 1 << i
                dp[state_only_i_visited][i] = (weights[start_node][i], [i])

            # list of submitted tasks
            reqs = []
            concurrent_layer = 0
            # number of concurrent tasks
            concurrent_tasks = 1
            for mask in range(1 << n):
                req: Future = executor.submit(
                    travel,
                    current_mask=mask,
                    dp=dp,
                    weights=weights
                )
                req.add_done_callback(partial(callback, mask=mask))
                reqs.append(req)
                concurrent_tasks -= 1
                if concurrent_tasks <= 0:
                    # wait for all submitted tasks to complete
                    # before starting the next batch of concurrent tasks
                    futures.wait(reqs)
                    reqs = []
                    concurrent_layer += 1
                    if concurrent_layer <= n:
                        # pascal triangle number
                        concurrent_tasks = math.factorial(n) // (
                                    math.factorial(concurrent_layer) * math.factorial(n - concurrent_layer)
                        )
                    else:
                        concurrent_tasks = 1

            futures.wait(reqs)

            print(
                f'\nFinish TSP with result {dp[-1][0]}\n'
                f'time taken to process {len(weights)} nodes TSP '
                f'with {size} processor: '
                f'{time.time() - start_time}\n'
                f'*****************************************************************************************************'
            )
            assert expected == dp[-1][0][0]
