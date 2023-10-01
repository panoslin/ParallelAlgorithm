# 502 Parallel Algorithm

## 1. Source codes & Time Complexity Analysis

### 1. [Serial TSP](tsp/tsp_serial_iterative.py)

Reconstructed from [this Leetcode post](https://leetcode.com/discuss/general-discussion/1125779) by [DBabichev](https://leetcode.com/DBabichev) which solves the problem of [Find the Shortest Superstring](https://leetcode.com/problems/find-the-shortest-superstring/). Under the hood, it's actuall a Travelling Salesman Problem. 

**1. Source code**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 9/28/23
# IDE: PyCharm
from typing import List, Tuple
from itertools import permutations


class TSP:
    @staticmethod
    def travel(weights: List[List[int]]):
        start_node = 0
        n = len(weights)

        # dp[state][node] representing the minimal cost/path,
        # visiting all nodes from bit mask `state` and last visited node is `node`
        dp: List[List[Tuple[float, List[int]]]] = [
            [(float('inf'), [])] * n
            for _ in range(1 << n)
        ]
        for i in range(n):
            # first node taken
            # should be after the start node
            state_only_i_visited = 1 << i
            dp[state_only_i_visited][i] = (weights[start_node][i], [i])

        for current_mask in range(1 << n):
            nodes_to_be_visited = [j for j in range(n) if current_mask & (1 << j)]

            # compare all possible permutation of nodes
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

        # all nodes visited and back to the start node
        return dp[(1 << n) - 1][0]

```



**2. Time Complexity**:

The implementation is identical to the Java solution provided in the assignment by [williamfiset](https://github.com/williamfiset/algorithms). The overall Time Complexity is `O(n^2 * 2^n)`:

Assuming the number of nodes are `n`. 

We first initialize a 2-D matrix `dp[state][node]` representing the minimal cost/path visiting all nodes from bit mask `state` and last visited node `node`. 

Then we construct the `dp` row by row starting from `dp[0]` to `dp[(1 << n) - 1]`. Inside of each row `current_mask`, we will construct it by reusing all the previous row `state_dest_not_visited` which with Hamming Distance of 1 comparing to `current_mask`. 

At last, we will return `dp[(1 << n) - 1][0]` which mean all nodes have been visited and last visited node is `0`, i.e, visiting all `n` nodes and back to the starting node.

Hence, the time complexity, to construct the `dp` matrix, is number of possible bit mask by number of permutation of any 2 nodes, i.e.,  `O(n^2 * 2^n)`.



### 2. [Multi-threaded TSP](tsp/tsp_threading.py)

Reconstructed from [the iterative version](tsp/tsp_serial_iterative.py). Instead of solving the `dp` matrix row by row, we put each row's processing to a  `helper` function in a thread pool created by `ThreadPoolExecutor` with thread count of `p` = `min(os.cpu_count(), len(weights))`. Inside of each thread, it will call `helper` function which update the `dp` matrix in the share memory for each row.

**1. Source  code**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by panos on 9/28/23
# IDE: PyCharm
import concurrent.futures
import os
from itertools import permutations
from typing import List, Tuple


class TSP:
    def __init__(self, weights: List[List[int]]):
        self.weights = weights
        self.thread_count = min(os.cpu_count(), len(weights))

    def travel(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            start_node = 0
            n = len(self.weights)

            # dp[state][node] representing the minimal cost/path,
            # visiting all nodes from bit mask `state` and last visited node is `node`
            dp: List[List[Tuple[float, List[int]]]] = [
                [(float('inf'), [])] * n
                for _ in range(1 << n)
            ]
            for i in range(n):
                # first node taken
                # should be after the start node
                state_only_i_visited = 1 << i
                dp[state_only_i_visited][i] = (self.weights[start_node][i], [i])

            # Start the load operations and mark each future with its parameters
            future_to_idx = {
                executor.submit(
                    self.helper,
                    current_mask=mask,
                    dp=dp,
                ): mask
                for mask in range(1 << n)
            }
            concurrent.futures.wait(future_to_idx)

            return dp[(1 << n) - 1][0]

    def helper(self, current_mask, dp):
        n = len(self.weights)
        nodes_to_be_visited = [j for j in range(n) if current_mask & (1 << j)]
        for dest, src in permutations(nodes_to_be_visited, 2):
            mask_dest_not_visited = current_mask ^ (1 << dest)
            dp[current_mask][dest] = min(
                dp[current_mask][dest],
                (
                    dp[mask_dest_not_visited][src][0] + self.weights[src][dest],
                    dp[mask_dest_not_visited][src][1] + [dest]
                ),
                key=lambda x: x[0]
            )

```



**2. Time Complexity**:

Instead of processing each row in sequential, we solve a `p` number of sub-problems in parallel. Updating the `dp` matrix is happening in shared memory, hence no extra communication overhead needed. Hence the overall Time Complexity is `O(n^2 * 2^n / p)`, when `p == 2^n`, the time complexity becomes `O(2^n)`

### 3. [MPI-based TSP](tsp/tsp_mpi.py)

This is a super naive MPI implementation. We have a master process to distribute sub-task to each sub-process and each of them will solve the task independently and return a list of result. 

Similar as the [thread version](tsp/tsp_threading.py), Instead of putting tasks into a `ThreadPoolExecutor`, we put them into a `MPIPoolExecutor` with processor count of p = nC<sub>n/2</sub>  , which essentially process the given `travel` function in a separate process. 

**1. Source code**

```python
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
            )
            assert expected == dp[-1][0][0]

```

**3 .Time Complexity**:

Similar as the thread version, we solve a `p` number of sub-problems in parallel. However, we don't have shared memory in separate processes, each time we call a `travel` sub-task will need to communicate variables (line 77) `mask`, `dp`, `weight` to sub-process. and returning (line 37) `result` as an array. 

Assuming:

- time to prepare to send a message t<sub>s</sub> 

- time to send a message is t<sub>w</sub>

- number of node is n

- message size m

	- send msg (line 77) size m<sub>1</sub> = 1 + 2<sup>n</sup> * n + n<sup>2</sup>
	- receive msg (line 37) size m<sub>2</sub> = n

- number of processor is p

	

The Time Complexity for each part is:

- $T_{send msg to sub process and run TSP}$ T<sub>send msg to sub process</sub> =  t<sub>s</sub> + t<sub>w</sub>*m<sub>1</sub>
- $T_{send msg to sub process and run TSP}$ T<sub>travel</sub> =  n<sup>2</sup>
- $T_{send msg to sub process and run TSP}$ T<sub>recv msg in master process</sub> = t<sub>s</sub> + t<sub>w</sub>*m<sub>2</sub> 
- $T_{send msg to sub process and run TSP}$ T<sub>stitch results</sub> = n



Hence for each iteration Time Comlexity is :

T<sub>iteration</sub> = T<sub>send msg to sub process</sub> + T<sub>travel</sub> +   T<sub>recv msg in master process</sub>

   = t<sub>s</sub> + t<sub>w</sub>*m<sub>1</sub> + n<sup>2</sup> +  t<sub>s</sub> + t<sub>w</sub>*m<sub>2</sub> 



The overall Time Complexity:

Instead of using a constant number of processor for each iteration, according to [the task dependency graph](#graph) of the problem, we can set a dynamic number of concurrency (line 75 and line 94). 

The number of prcessors needed for each iteration is similar to a Pascal Triangle. For example, for a n=4 TSP problem, we can solve the proble in n + 1 = 5 steps, each step will need a concurrency of 1, 4, 6, 4, 1, i.e., nC0, nC1, nC2, nC3, nC4.

The average degree of concurrency is `total number of tasks / critical path length = n ^ 2 / (n + 1) = 16/5 = 3.2` 

Assuming the number of processor can be `p` == nC<sub>n/2</sub>, the over Time Complexity can be:

T = T<sub>iteration</sub> * (n + 1) + T<sub>stitch results</sub> = (2t<sub>s</sub> + (1 + 2<sup>n</sup> n + n<sup>2</sup> + n)t<sub>w</sub> + n<sup>2</sup>) n + n

### 4. [Test cases](tsp/testcases.py)

A list of test cases in [this file](tsp/testcases.py)

## 2. Results and Conclusions

<p id=graph>Task Dependency Graph of MPI version of TSP</a>:

![Pallel algo graph](/Users/linguohui/Projects/ParallelAlgorithm/README.assets/Pallel algo graph.png)

Performance Comparison

| n    | Serial  | Threaded Version | Thread Count | MPI Version | processors | Speedup | efficiency |
| ---- | ------- | ---------------- | ------------ | ----------- | ---------- | ------- | ---------- |
| 3    | 0.0001  | 0.0032           | 8            | 0.0189      | 4          | 0.0027  | 0.0007     |
| 4    | 0.0001  | 0.0058           | 16           | 0.0119      | 4          | 0.0048  | 0.0012     |
| 5    | 0.0001  | 0.0060           | 32           | 0.0252      | 4          | 0.0057  | 0.0014     |
| 6    | 0.0004  | 0.0125           | 64           | 0.0602      | 4          | 0.0066  | 0.0017     |
| 7    | 0.0011  | 0.0178           | 128          | 0.1452      | 4          | 0.0074  | 0.0018     |
| 8    | 0.0029  | 0.0305           | 256          | 0.4286      | 4          | 0.0068  | 0.0017     |
| 9    | 0.0073  | 0.0595           | 512          | 1.8386      | 4          | 0.0040  | 0.0010     |
| 10   | 0.0162  | 0.1145           | 1,024        | 8.4510      | 4          | 0.0019  | 0.0005     |
| 11   | 0.0368  | 0.2437           | 2,048        | 30.9026     | 4          | 0.0012  | 0.0003     |
| 12   | 0.0905  | 0.4797           | 4,096        | 104.4777    | 4          | 0.0009  | 0.0002     |
| 13   | 0.2176  | 1.0923           | 8,192        | 382.0221    | 4          | 0.0006  | 0.0001     |
| 14   | 0.4929  | 2.6333           | 16,384       | NA          | NA         | NA      | NA         |
| 15   | 1.1630  | 7.5699           | 32,768       | NA          | NA         | NA      | NA         |
| 16   | 2.6974  | 19.1027          | 65,536       | NA          | NA         | NA      | NA         |
| 17   | 6.2469  | 40.4822          | 131,072      | NA          | NA         | NA      | NA         |
| 18   | 14.6932 | 99.9603          | 262,144      | NA          | NA         | NA      | NA         |
| 19   | 31.0961 | 216.1901         | 524,288      | NA          | NA         | NA      | NA         |
| 20   | 85.9556 | 495.3688         | 1,048,576    | NA          | NA         | NA      | NA         |