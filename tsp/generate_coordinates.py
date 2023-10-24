#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 10/6/23
# IDE: PyCharm
import random
from typing import List


def generate_coordinates(
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
            distance = calculate_distance(coordinates[point1_idx], coordinates[point2_idx])
            adjacency_matrix[point1_idx][point2_idx] = distance
    return adjacency_matrix


def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """
    calculate distance
    :param point1:
    :param point2:
    :return:
    """
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


if __name__ == '__main__':
    from pprint import pprint

    _ = generate_coordinates(4, 0, 1, 0, 1)
    pprint(_)
    _ = generate_adjacency_matrix(_)
    pprint(_)
