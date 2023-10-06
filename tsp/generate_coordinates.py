#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 10/6/23
# IDE: PyCharm
import random
from typing import List


def generate(n: int, x1, x2, y1, y2) -> List[List[int]]:
    """
    generate n pairs of coordinates
    :param n:
    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :return:
    """
    coordinates = []
    for _ in range(n):
        x = random.uniform(x1, x2)
        y = random.uniform(y1, y2)
        coordinates.append([x, y])
    return coordinates


if __name__ == '__main__':
    _ = generate(4, 0, 1, 0, 1)
    print(_)
