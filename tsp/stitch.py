#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 10/22/23
# IDE: PyCharm
from typing import List

from generate_coordinates import calculate_distance


def stitch2tsp(
        coordinates_left: List[List[float]],
        coordinates_right: List[List[float]],
        tsp_path_left: List[int],
        tsp_path_right: List[int],
        min_cost_left: float,
        min_cost_right: float
) -> List[int]:
    """
    Stitch 2 Euclidean Traveling Salesperson Problem

    returning the minimal cost/path for traveling all nodes in
    distance_matrix_left and distance_matrix_right

    :param coordinates_left: list of coordinates of a node from left
    :param coordinates_right: list of coordinates of a node from right
    :param tsp_path_left: a path travelling all node in distance_matrix_left with minimal cost
    :param tsp_path_right: a path travelling all node in distance_matrix_right with minimal cost
    :param min_cost_left: the minimal cost for traveling all nodes in distance_matrix_left
    :param min_cost_right: the minimal cost for traveling all nodes in distance_matrix_right
    :return: a path travelling all node in distance_matrix_left and distance_matrix_right with minimal cost
    """
    n, m = len(tsp_path_left), len(tsp_path_right)

    # indices of the parent nodes of the swapped edge
    # from the 2 tsp paths respectively
    min_swap = [0, 0]

    # minimum cost after optimal swapping
    min_cost = float('inf')

    # try every edges
    for left_parent_node_idx in range(n):
        for right_parent_node_idx in range(m):
            left_child_node_idx = (left_parent_node_idx + 1) % n
            right_child_node_idx = (right_parent_node_idx + 1) % m

            left_parent_node = tsp_path_left[left_parent_node_idx]
            right_parent_node = tsp_path_right[right_parent_node_idx]
            left_child_node = tsp_path_left[left_child_node_idx]
            right_child_node = tsp_path_right[right_child_node_idx]

            new_cost_diff = swap_cost(
                coordinates_left,
                coordinates_right,
                left_parent_node,
                right_parent_node,
                left_child_node,
                right_child_node,
            )
            if new_cost_diff < min_cost:
                min_swap = [left_child_node_idx, right_child_node_idx]
                min_cost = new_cost_diff

    left_child_node_idx = min_swap[0]
    right_child_node_idx = min_swap[1]
    # increment the indices of the right path
    # to help identify
    tsp_path_right = [ele + n for ele in tsp_path_right]
    merged_path = (
            tsp_path_left[:left_child_node_idx] +
            tsp_path_right[right_child_node_idx:] +
            tsp_path_right[:right_child_node_idx] +
            tsp_path_left[left_child_node_idx:]
    )
    merged_cost = min_cost_left + min_cost_right + min_cost
    return merged_cost, merged_path


def swap_cost(
        coordinates_left: List[List[float]],
        coordinates_right: List[List[float]],
        left_parent_node: int,
        right_parent_node: int,
        left_child_node: int,
        right_child_node: int
) -> float:
    # sum of current edges
    current_cost = (
            calculate_distance(coordinates_left[left_parent_node], coordinates_left[left_child_node], ) +
            calculate_distance(coordinates_right[right_parent_node], coordinates_right[right_child_node])
    )

    # cost of swapping
    distance_left_parent_to_right_child = calculate_distance(
        coordinates_left[left_parent_node],
        coordinates_right[right_child_node],
    )
    distance_right_parent_to_left_child = calculate_distance(
        coordinates_right[right_parent_node],
        coordinates_left[left_child_node],
    )
    # total cost of swapping
    total_cost = distance_left_parent_to_right_child + distance_right_parent_to_left_child
    # difference in cost
    cost_diff = total_cost - current_cost
    return cost_diff


if __name__ == '__main__':
    from generate_coordinates import generate_coordinates, generate_adjacency_matrix
    from tsp_serial_iterative import TSP
    from pprint import pprint

    # coordinates_left = [[0, 0], [0, 1]]
    # coordinates_right = [[1, 0], [1, 1]]

    # coordinates_left = [[0.052238614457627786, 0.8277863742033538], [0.658728565577255, 0.492198794069551], [0.834736624412124, 0.19340830645368856]]
    # coordinates_right = [[1.9361984043619007, 1.9434814361695623], [1.6324620840022, 1.1891255255805668], [1.4595302614432137, 1.259869055785428]]

    coordinates_left = generate_coordinates(3, 0, 1, 0, 1)
    coordinates_right = generate_coordinates(3, 1, 2, 1, 2)
    print(f"{coordinates_left=}")
    print(f"{coordinates_right=}")
    distance_matrix_left = generate_adjacency_matrix(coordinates_left)
    distance_matrix_right = generate_adjacency_matrix(coordinates_right)
    min_cost_left, tsp_path_left = TSP.travel(distance_matrix_left)
    pprint(["TSP left", tsp_path_left])
    min_cost_right, tsp_path_right = TSP.travel(distance_matrix_right)
    pprint(["TSP right", [ele + len(coordinates_left) for ele in tsp_path_right]])
    _ = stitch2tsp(
        coordinates_left,
        coordinates_right,
        tsp_path_left,
        tsp_path_right,
        min_cost_left,
        min_cost_right
    )
    pprint(["Stitched path", _])
    merged_coordinates = coordinates_left + coordinates_right
    merged_distance_matrix = generate_adjacency_matrix(merged_coordinates)
    expected_result = TSP.travel(merged_distance_matrix)
    pprint(["Expected path", expected_result])
