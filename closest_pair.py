import multiprocessing
from matplotlib import pyplot as plt
import numpy as np


def merge_by_y(left, right):
    """
    Implements merge of 2 sorted lists by the second coordinate
    :param args: support explicit left/right args, as well as a two-item
                tuple which works more cleanly with multiprocessing.
    :return: merged list
    """
    left_length, right_length = len(left), len(right)
    left_index, right_index = 0, 0
    merged = []
    while left_index < left_length and right_index < right_length:
        if left[left_index, 1] <= right[right_index, 1]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1
    if left_index == left_length:
        merged.extend(right[right_index:])
    else:
        merged.extend(left[left_index:])
    return merged


# euclidean distance calculation
def distance(p1, p2):
    return np.math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Brute force method to calculate distance
def _brute_closest_pair_finder(X):
    min_dist = distance(X[0], X[1])
    p1 = X[0]
    p2 = X[1]
    len_X = len(X)
    if len_X == 2:
        return p1, p2, min_dist
    for i in range(len_X - 1):
        for j in range(i + 1, len_X):
            d = distance(X[i, :], X[j, :])
            if d < min_dist:  # Update min_dist and points
                min_dist = d
                p1, p2 = X[i, :], X[j, :]
    return p1, p2, min_dist


def _boundary_merge(X, distances, point_pairs, xm):
    """
    Finds the closed pair in a point set merged from 2 parts of recursion
    :param X: points merged and sorted by y (axis = 1)
    :param distances: smallest distances found of in left and right parts of the input
    :param point_pairs: pairs of points which correspond to distances
    :param xm: median by x (axis = 0)
    :return: pair of closest points and distance between them
    """

    min_d = min(distances)  # min_d is minimum distance so far
    M_ind = np.where((X[:, 0] >= (xm - min_d)) & (X[:, 0] <= (xm + min_d)))  # pair_with_min_d = point_pairs[distances.index(d)]
    # print(X.shape)
    # print("M",M_ind)
    M = X[M_ind]
    # print(M)
    p1, p2, d_M = _brute_closest_pair_finder(M)  # d_M is minimum distance found on boundary
    if d_M not in distances:
        distances.append(d_M)
        point_pairs.append((p1, p2))
        print("Point pairs after boundary merge\n", point_pairs)
    else:
        print("The minimum distance is not on boundary.")

    min_d = min(distances)
    pair_with_min_d = point_pairs[distances.index(min_d)]
    print("Min distance on this step\n", min_d)
    print("Pair with min distance on this step\n", pair_with_min_d)
    return pair_with_min_d[0], pair_with_min_d[1], min_d

# generate a process number which will be index in dictionary
def generate_process_number(process_id):
    return int(hash(process_id) % 1e8)


def sort_by_y(points):
    ind = np.argsort(points[:, 1])
    return points[ind]


def closest_pair(points, return_dict=None, verbose=False):
    if len(points) <= MIN_SIZE_OF_ARRAY_PER_THREAD:
        print(multiprocessing.current_process())
        n = generate_process_number(multiprocessing.current_process())
        return_dict[n] = (sort_by_y(points), _brute_closest_pair_finder(points))
    else:
        x_median = medianSearch(points[:, 0])
        if verbose:
            print("Median on this step", x_median)
        left = points[np.where(points[:, 0] < x_median)]
        right = points[np.where(points[:, 0] >= x_median)]
        jobs = []
        manager = multiprocessing.Manager()
        return_dict_input = manager.dict()
        for data in [left, right]:
            jobs.append(multiprocessing.Process(target=closest_pair, args=(data, return_dict_input)))
        for job in jobs:
            job.start()
        if verbose:
            print(multiprocessing.current_process())
        for job in jobs: job.join()
        for job in jobs: job.terminate()
        results = return_dict_input.values()
        res_len = len(results)
        merged = np.array(merge_by_y(results[0][0], results[1][0]))
        distances = [results[i][1][2] for i in range(res_len)]
        point_pairs = [(results[i][1][0], results[i][1][1]) for i in range(res_len)]
        if verbose:
            print("\nResult of 2 parallel task execution")
            print("Current shape of merged points", merged.shape)
            print("Min distances found by each of tasks\n", distances)
            print("Point pairs after  merge of tasks\n", point_pairs)
        res_boundary_merge = _boundary_merge(merged, distances, point_pairs, x_median)
        n = generate_process_number(multiprocessing.current_process())
        if return_dict is None: return_dict = manager.dict()
        return_dict[n] = (merged, res_boundary_merge)

        return res_boundary_merge

# calculates median value in a list
def medianSearch(list):
    return np.median(list)

# write integer data to csv file
def write_to_file(data, file_name):
    np.savetxt(file_name + ".csv", data, fmt='%d', delimiter=",")

# creating scatterplot of a 2d dataset
def plot_points(data, plot_name, show=False):
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(plot_name)
    plt.savefig(plot_name + ".png")
    if show:
        plt.show()


def min_size_of_array(full_input_len):
    return full_input_len / NUM_WORKERS


NUM_WORKERS = 4
ARRAY_SIZE = 2 ** 5
MIN_SIZE_OF_ARRAY_PER_THREAD = ARRAY_SIZE / NUM_WORKERS

if __name__ == "__main__":
    np.random.seed(123)
    input_points = (np.random.randn(ARRAY_SIZE, 2) * 100).astype(int)
    write_to_file(input_points, "input")
    plot_points(input_points, "input", False)
    result = closest_pair(input_points, verbose=True)
    print("\n\nRESULT: \nThe closed pair:{0} and {1}\nDistance: {2:.5f}".format(result[0], result[1], result[2]))
