# closest pairs by divide and conquer
# David Eppstein, UC Irvine, 7 Mar 2002

from __future__ import generators
import multiprocessing
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt, pow

def closestpair(L):
    def distance(p, q):
        return sqrt(pow(p[0] - q[0], 2) + pow(p[1] - q[1], 2))


    # check whether pair (p,q) forms a closer pair than one seen already
    def testpair(p, q):
        d = distance(p, q)
        if d < best[0]:
            best[0] = d
            best[1] = p, q

    # merge two sorted lists by y-coordinate
    def merge_by_y(A, B):
        i = 0
        j = 0
        while i < len(A) or j < len(B):
            if j >= len(B) or (i < len(A) and A[i][1] <= B[j][1]):
                yield A[i]
                i += 1
            else:
                yield B[j]
                j += 1

    # Find closest pair recursively; returns all points sorted by y coordinate
    def recur(L):
        if len(L) < 2:
            return L
        split = len(L) // 2
        splitx = L[split,1]
        L = list(merge_by_y(recur(L[:split]), recur(L[split:])))

        # Find possible closest pair across split line - boundary merge
        # Here I compare the new minimum distances found with a global minimum so far
        # This change reduces the size of E, speeding up the algorithm a little.
        E = [p for p in L if abs(p[0] - splitx) < best[0]]
        for i in range(len(E)):
            for j in range(1, 8):
                if i + j < len(E):
                    testpair(E[i], E[i + j])
        return L

    def sort_by_column(points, column):
        ind = np.argsort(points[:, column])
        return points[ind]

    L = sort_by_column(L,0)
    # Use L[0],L[1] as the initial guess of the distance.
    best = [distance(L[0], L[1]), (L[0], L[1])]
    recur(L)
    return best


def parallel_closest_pair(points):
    p = multiprocessing.Pool(NUM_WORKERS)
    data_chunks = np.split(points, NUM_WORKERS)
    results = p.map(closestpair, data_chunks)
    # points_returned = [results[i][1] for i in range(NUM_WORKERS)]
    p.close()
    return results


NUM_WORKERS = 4
ARRAY_SIZE = 2 ** 4

if __name__ == "__main__":
    np.random.seed(123)
    input_points = (np.random.randn(ARRAY_SIZE, 2) * 100).astype(int)
    print("Input points:\n",input_points)
    result_sequential = closestpair(input_points)
    result_parallel = parallel_closest_pair(input_points)
    print("Sequential result:\n", result_sequential)
    print("Parallel result:\n", result_parallel)
