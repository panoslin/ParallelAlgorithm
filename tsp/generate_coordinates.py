#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 10/6/23
# IDE: PyCharm
import random
from typing import List


def generate(
        n: int,
        x_axis_left_bound: int,
        x_axis_right_bound: int,
        y_axis_left_bound: int,
        y_axis_right_bound: int
) -> List[List[float]]:
    """
    generate n pairs of coordinates
    :param n:
    :param x_axis_left_bound:
    :param x_axis_right_bound:
    :param y_axis_left_bound:
    :param y_axis_right_bound:
    :return:
    """
    coordinates = []
    for _ in range(n):
        x = random.uniform(x_axis_left_bound, x_axis_right_bound)
        y = random.uniform(y_axis_left_bound, y_axis_right_bound)
        coordinates.append([x, y])
    return coordinates


def generate_adjacency_matrix(coordinates: List[List[float]]) -> List[List[float]]:
    """
    generate adjacency matrix
    :param coordinates:
    :return:
    """
    adjacency_matrix = [[0 for _ in range(len(coordinates))] for _ in range(len(coordinates))]
    for point1_idx in range(len(coordinates)):
        for point2_idx in range(len(coordinates)):
            distance = (
                               (coordinates[point1_idx][0] - coordinates[point2_idx][0]) ** 2 +
                               (coordinates[point1_idx][1] - coordinates[point2_idx][1]) ** 2
                       ) ** 0.5
            adjacency_matrix[point1_idx][point2_idx] = distance
    return adjacency_matrix


if __name__ == '__main__':
    from pprint import pprint

    _ = generate(4, 0, 1, 0, 1)
    pprint(_)
    _ = generate_adjacency_matrix(_)
    pprint(_)
