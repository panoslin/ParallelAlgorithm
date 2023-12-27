#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 9/28/23
# IDE: PyCharm
"""
mpiexec -np 4 python tsp/tsp_mpi_pipeline.py
"""
import time
from collections import defaultdict
from typing import List, Tuple, Iterable, Dict

import numpy as np
from mpi4py import MPI


def compute_tsp(
        node_count: int,
        weights: List[List[int]],
        processor_count: int,
        comm: MPI.Comm
):
    # queue storing any intermediate dp result of each bit mask
    mask_queue: Dict[int: List] = defaultdict(list)
    while True:
        # 1. recv dependency task
        data = np.empty(node_count + 3, dtype='int32')
        comm.Recv([data, MPI.INT], source=MPI.ANY_SOURCE)
        cost, mask, number_visited_node, path = deserialize_data(data)

        # final result recv at rank 0:
        if mask == -1:
            return cost, path

        mask_queue[mask].append((path, cost))

        # indicator of all the cells of the dp column has been filled
        is_all_subtask_done = len(mask_queue[mask]) == number_visited_node

        # all dependent tasks completed
        if is_all_subtask_done:
            if mask == (1 << node_count) - 2:
                # send termination message to all processors
                optimal_cost, optimal_path = find_optimal_path(mask_queue[mask], 0, weights)
                data[-1] = optimal_cost
                for i in range(processor_count):
                    data[-2] = -1
                    comm.Send([data, MPI.INT], dest=i)

                continue

            # find the min cost path to each unvisited node
            # according to all the subtasks gathered
            for node in range(1, node_count):
                # not visited node
                if mask & (1 << node) == 0:
                    next_mask = mask | (1 << node)
                    next_processor = (next_mask + 1) // 2 % processor_count
                    optimal_cost, optimal_path = find_optimal_path(mask_queue[mask], node, weights)
                    data[number_visited_node] = node
                    data[-3] = number_visited_node + 1
                    data[-2] = next_mask
                    data[-1] = optimal_cost
                    comm.Send([data, MPI.INT], dest=next_processor)


def find_optimal_path(exist_path: List[Tuple[List, int]], next_node, weights):
    """
    find the path with min total cost from exist path's last node to next_node
    :param exist_path:
    :param next_node:
    :param weights:
    :return:
    """
    optimal_path, optimal_cost = min(
        exist_path,
        key=lambda path_and_cost: path_and_cost[1] + weights[path_and_cost[0][-1]][next_node]
    )
    return optimal_cost + weights[optimal_path[-1]][next_node], optimal_path + [next_node]


def deserialize_data(data):
    """
    deserialize sequence of array receiving from MPI.Recv
    :param data:
    :return:
    """
    cost = data[-1]
    mask = data[-2]
    number_visited_node = data[-3]
    path: Iterable[int] = data[:number_visited_node]
    return cost, mask, number_visited_node, path


def main():
    from testcases import testcases

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    processor_count = comm.Get_size()

    for weights, expected_cost in testcases:
        start_time = time.time()
        if rank == 0:
            # 1. One to All broadcast: adjacency matrix
            comm.bcast(weights, root=0)
            node_count = len(weights)

            # 2. initiate subtasks: traverse all nodes starting from node 0
            path = [-1] * node_count
            number_visited_node = 1
            for node in range(1, node_count):
                mask = 1 << node
                path[0] = node
                dest_process = (mask + 1) // 2 % processor_count
                data = np.array(path + [number_visited_node, mask, weights[0][node]], dtype='i')
                comm.Send([data, MPI.INT], dest=dest_process)
        else:
            # 1. recv weights: One to All Broadcast from rank 0
            weights = comm.bcast(None, root=0)
            node_count = len(weights)

        # loop compute tsp
        optimal_cost, optimal_path = compute_tsp(
            node_count=node_count,
            weights=weights,
            processor_count=processor_count,
            comm=comm,
        )
        if rank == 0:
            # return final result
            print(
                f'*****************************************************************************************************'
                f'\nFinish TSP with result {optimal_cost}\n'
                f'time taken to process {len(weights)} nodes TSP '
                f'with {processor_count} processor: '
                f'{time.time() - start_time}\n'
                f'{optimal_path=}\n'
                f'*****************************************************************************************************'
            )
            assert expected_cost == optimal_cost


if __name__ == '__main__':
    main()
