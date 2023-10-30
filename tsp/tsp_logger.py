#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 10/27/23
# IDE: PyCharm
import logging
from logging import Logger


class TSPLogger(Logger):
    def info(self, msg, rank, *args, **kwargs) -> None:
        msg = f"Process {rank}: {msg}"
        super().info(msg, *args, **kwargs)

    def debug(self, msg, rank, *args, **kwargs) -> None:
        msg = f"Process {rank}: {msg}"
        super().debug(msg, *args, **kwargs)

    def critical(self, msg, rank, *args, **kwargs) -> None:
        msg = f"Process {rank}: {msg}"
        super().critical(msg, *args, **kwargs)


def tsp_logger():
    """
    Create a custom logger with a specific logging format.

    Returns:
        logging.Logger: A custom logger object.
    """
    logger = TSPLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(levelname)s- %(asctime)s %(message)s')

    # file_handler = logging.FileHandler('tsp.log')
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    # stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


if __name__ == '__main__':
    # Example usage
    logger = tsp_logger()
    logger.info('This is an info message', 1)
    logger.debug('This is a debug message', 1)
