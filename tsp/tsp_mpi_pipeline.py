#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 9/28/23
# IDE: PyCharm
"""
mpiexec -np 4 python tsp/tsp_mpi_pipeline.py
"""
import traceback
from typing import List, Tuple

import numpy as np
from mpi4py import MPI

from tsp_logger import tsp_logger


def error_catching(*args, **kwargs):
    try:
        return compute_tsp(*args, **kwargs)
    except Exception as e:
        logger.critical(traceback.format_exc(), kwargs['rank'])
        raise e


def compute_tsp(major_task_count, rank, task_queue):
    # 3. loop compute tsp
    while major_task_count > 0:
        # 4. recv dependency task
        logger.debug(f"receiving ...", rank)

        data = np.empty(node_count + 3, dtype='i')
        comm.Recv(
            [data, MPI.INT],
            source=MPI.ANY_SOURCE,
            tag=1
        )
        cost = data[-1]
        mask = data[-2]
        number_visited_node = data[-3]
        path = data[:number_visited_node]
        # mask, path, cost = comm.recv()
        logger.debug(f"recv task {mask:0{node_count}b} {(path, cost)}...", rank)

        task_idx = (mask - 1) // (processor_count * 2)
        task_queue[task_idx].append((mask, path, cost))
        logger.debug(f"appended task {mask:0{node_count}b} {(path, cost)} to queue ...", rank)
        logger.debug(f"{locals()=}", rank)

        is_all_subtask_done = (len(task_queue[task_idx]) == number_visited_node) or (mask == (1 << node_count) - 1)
        if is_all_subtask_done:
            major_task_count -= 1
            logger.debug(f"mask {mask:0{node_count}b}: finish collecting dependent tasks", rank)
            # all dependent tasks completed
            # start the current task
            if mask == (1 << node_count) - 1:
                # final mask
                # send result back to rank 0
                logger.info(
                    f"task {mask} "
                    f"{data=} "
                    f"sent to process 0",
                    rank
                )
                comm.Isend(
                    [data, MPI.INT],
                    dest=0,
                    tag=2,
                )

            else:
                # not final mask
                # find the min cost path to each unvisited node

                if mask == (1 << node_count) - 2:
                    starting_node = 0
                else:
                    starting_node = 1

                for node in range(starting_node, node_count):
                    # not visited node
                    if not_visited(mask, node):
                        next_mask = mask | (1 << node)
                        next_processor = (next_mask + 1) // 2 % processor_count
                        optimal_cost, optimal_path = find_optimal_path(task_queue[task_idx], node)
                        # data = np.array(
                        #     [optimal_path + [number_visited_node + 1, next_mask, optimal_cost]],
                        #     'i'
                        # )
                        data[number_visited_node] = node
                        data[-3] = number_visited_node + 1
                        data[-2] = next_mask
                        data[-1] = optimal_cost
                        comm.Isend(
                            [data, MPI.INT],
                            dest=next_processor,
                            tag=1,
                        )
                        logger.debug(
                            f"task {next_mask:0{node_count}b} "
                            f"{data=} "
                            f"sent to process {next_processor}",
                            rank
                        )
    else:
        logger.info(f"finish all tasks", rank)


def not_visited(mask: int, node: int):
    return mask & (1 << node) == 0


def find_optimal_path(exist_path: List[Tuple[int, List, int]], next_node):
    """

    :param exist_path:
        task_queue[task_idx].append((mask, path, cost))
    :param next_node:
    :return:
    """
    _, optimal_path, optimal_cost = min(
        exist_path,
        key=lambda x: x[2] + weights[x[1][-1]][next_node]
    )
    return optimal_cost + weights[optimal_path[-1]][next_node], optimal_path + [next_node]


if __name__ == '__main__':
    from testcases import testcases
    import time

    logger = tsp_logger()
    # init MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    processor_count = comm.Get_size()

    if rank == 0:
        # for weights, expected_cost in testcases:
        weights: List[List[int]] = testcases[12][0]
        expected_cost: int = testcases[12][1]

        start_time = time.time()
        # weights = np.array(weights)
        # 1. One to All broadcast: adjacency matrix
        comm.bcast(weights, root=0)
        logger.info(f"recv matrix {weights[0]}...", rank)
        node_count = len(weights)
        # 2. initiate subtasks: traverse all nodes starting from node 0
        path = [-1] * node_count
        for node in range(1, node_count):
            mask = 1 << node
            path[0] = node
            cost = weights[0][node]
            number_visited_node = 1
            dest_process = (mask + 1) // 2 % processor_count
            logger.info(f"init task {mask:0{node_count}b} {(path, cost)} sent to process {dest_process}", rank)
            data = np.array([path + [number_visited_node, mask, cost]], dtype='i')
            comm.Isend([data, MPI.INT], dest=dest_process, tag=1)

        # 3. init task count integer
        task_count = 2 ** (node_count - 1) // processor_count
        step = 2 * processor_count
        max_mask = (1 << node_count)
        first_mask = rank * 2
        task_count = (max_mask - first_mask) // step + (rank != 0 and rank * 2 < max_mask)
        task_queue = [[] for _ in range(task_count)]

        # 4.  loop compute tsp
        error_catching(
            major_task_count=task_count,
            rank=rank,
            task_queue=task_queue
        )

        # 5. receive final result
        logger.info(f"pending for final task.", rank)
        data = np.empty(node_count + 3, dtype='i')
        comm.Recv(
            [data, MPI.INT],
            source=(1 << node_count) // 2 % processor_count,
            tag=2
        )
        logger.info(f"Recv {data=}", rank)
        optimal_cost = data[-1]
        optimal_path = data[:data[-3]]
        # 6. return final result
        logger.info(
            f'*****************************************************************************************************'
            f'\nFinish TSP with result {optimal_cost}\n'
            f'time taken to process {len(weights)} nodes TSP '
            f'with {processor_count} processor: '
            f'{time.time() - start_time}\n'
            f'{optimal_path=}\n'
            f'*****************************************************************************************************',
            rank
        )
        assert expected_cost == optimal_cost
        # return cost, path

    else:
        # 1. recv weights: One to All Broadcast
        weights = comm.bcast(None, root=0)
        logger.info(f"recv matrix {weights[0]}...", rank)
        # 2. init task count integer
        node_count = len(weights)
        task_count = 2 ** (node_count - 1) // processor_count
        step = 2 * processor_count
        max_mask = (1 << node_count)
        first_mask = rank * 2
        task_count = (max_mask - first_mask) // step + (rank != 0 and rank * 2 < max_mask)
        task_queue = [[] for _ in range(task_count)]
        # 3. loop compute tsp
        error_catching(major_task_count=task_count, rank=rank, task_queue=task_queue)
