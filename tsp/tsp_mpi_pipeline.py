#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 9/28/23
# IDE: PyCharm
"""
mpiexec -np 4 python tsp/tsp_mpi_pipeline.py
"""
import time
from collections import defaultdict
from typing import List, Tuple, Iterable

import numpy as np
from mpi4py import MPI


def compute_tsp(mask_count):
    # queue storing any intermediate dp result of each bit mask
    mask_queue = defaultdict(list)
    while mask_count > 0:
        # 1. recv dependency task
        data = np.empty(node_count + 3, dtype='int32')
        comm.Recv(
            [data, MPI.INT],
            source=MPI.ANY_SOURCE,
            tag=1
        )
        cost, mask, number_visited_node, path = deserialize_data(data)

        # final result recv at rank 0:
        if mask == (1 << node_count) - 1:
            return cost, path

        mask_queue[mask].append((path, cost))

        # indicator of all the cells of the dp column has been filled
        is_all_subtask_done = len(mask_queue[mask]) == number_visited_node

        # all dependent tasks completed
        if is_all_subtask_done:
            mask_count -= 1
            start_node = 0 if (mask == (1 << node_count) - 2) else 1
            # find the min cost path to each unvisited node
            # according to all the subtasks gathered
            for node in range(start_node, node_count):
                # not visited node
                if not_visited(mask, node):
                    next_mask = mask | (1 << node)
                    next_processor = mask_to_processor(next_mask)
                    optimal_cost, optimal_path = find_optimal_path(mask_queue[mask], node, weights)
                    data[number_visited_node] = node
                    data[-3] = number_visited_node + 1
                    data[-2] = next_mask
                    data[-1] = optimal_cost
                    comm.Isend(
                        [data, MPI.INT],
                        dest=next_processor,
                        tag=1,
                    )


def print_rank(*args):
    """
    for logging to stdout
    :param args:
    :return:
    """
    print(f"{rank=}: {' '.join(args)}")


def not_visited(mask: int, node: int):
    """
    is a node not visited
    :param mask:
    :param node:
    :return:
    """
    return mask & (1 << node) == 0


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


def mask_to_processor(mask):
    if mask == (1 << node_count) - 1:
        # final result should send to rank 0
        return 0
    else:
        # todo: seems like some issue when node larger than 13.
        # mapping - should be load balanced
        # return 1
        # return mask % processor_count
        return (mask + 1) // 2 % processor_count


def cal_task_count(rank):
    step = 2 * processor_count
    max_mask = (1 << node_count)
    first_mask = rank * 2
    task_count = (max_mask - first_mask) // step + (rank != 0 and rank * 2 < max_mask)
    return task_count


def main():
    # init MPI communicator
    global comm
    global rank
    global processor_count
    global weights
    global node_count

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    processor_count = comm.Get_size()

    if rank == 0:
        from testcases import testcases
        testcase = testcases[10]
        weights = testcase[0]
        expected_cost: int = testcase[1]

        start_time = time.time()

        # 1. One to All broadcast: adjacency matrix
        comm.bcast(weights, root=0)
        node_count = len(weights)

        # 2. initiate subtasks: traverse all nodes starting from node 0
        path = [-1] * node_count
        number_visited_node = 1
        for node in range(1, node_count):
            mask = 1 << node
            path[0] = node
            dest_process = mask_to_processor(mask)
            data = np.array(path + [number_visited_node, mask, weights[0][node]], dtype='i')
            print_rank(f"initial {mask=} sent to {dest_process=}")
            comm.Send(
                [data, MPI.INT],
                dest=dest_process,
                tag=1
            )

        # 3. init task count: number of mask each process should handle
        task_count = cal_task_count(rank)

        # 4. loop compute tsp
        optimal_cost, optimal_path = compute_tsp(
            mask_count=task_count,
        )

        # 5. return final result
        print_rank(
            f'*****************************************************************************************************'
            f'\nFinish TSP with result {optimal_cost}\n'
            f'time taken to process {len(weights)} nodes TSP '
            f'with {processor_count} processor: '
            f'{time.time() - start_time}\n'
            f'{optimal_path=}\n'
            f'*****************************************************************************************************'
        )
        assert expected_cost == optimal_cost
    else:
        # 1. recv weights: One to All Broadcast
        weights = comm.bcast(None, root=0)
        node_count = len(weights)
        print_rank(f"receive weights of length {len(weights)=}")

        # 2. init task count: number of mask each process should handle
        task_count = cal_task_count(rank)

        # 3. loop compute tsp
        compute_tsp(mask_count=task_count)


if __name__ == '__main__':
    main()
