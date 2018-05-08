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


def distance(p1, p2):
    """
    :return: euclidean distance
    """
    return np.math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Brute force method to calculate distance for n<=3
def _brute_closest_pair_finder(X):
    min_dist = distance(X[0], X[1])
    p1 = X[0]
    p2 = X[1]
    len_X = len(X)
    if len_X == 2:
        return p1, p2, min_dist
    for i in range(len_X - 1):
        for j in range(i + 1, len_X):
            if i != 0 and j != 1:
                d = distance(X[i, :], X[j, :])
                if d < min_dist:  # Update min_dist and points
                    min_dist = d
                    p1, p2 = X[i, :], X[j, :]
    return p1, p2, min_dist


def _boundary_merge(X, distances, xm):
    """
    Finds the closed distance between merged from several parts points
    :param X: merged by y points
    :param dist_l:
    :param xm: median
    :return:
    """
    # d is minimum distance so far
    d = min(distances)
    print(X)
    M_ind = np.where(X[:,0]>= (xm-d) & X[:,0] <= (xm+d))
    M = X[M_ind]

    #d_M is minimum distance found on boundary
    d_M = _brute_closest_pair_finder(M)
    return min(d, d_M)


def generate_process_number(process_id):
    return int(hash(process_id) % 1e8)


def sort_by_y(points):
    ind = np.argsort(points[:, 1])
    return points[ind]


def work(points, return_dict):
    if len(points) <= 4:
        print(multiprocessing.current_process())
        n = generate_process_number(multiprocessing.current_process())
        return_dict[n] = (sort_by_y(points), _brute_closest_pair_finder(points))
    else:
        x_median = medianSearch(points[:, 0])
        # print("median ", x_median)
        left = points[np.where(points[:, 0] < x_median)]
        right = points[np.where(points[:, 0] >= x_median)]
        jobs = []
        for data in [left, right]:
            jobs.append(multiprocessing.Process(target=work, args=(data, return_dict)))
        # print(multiprocessing.current_process(), return_dict.values(), "\n\n")
        for job in jobs: job.start()
        for job in jobs: job.join()
        for job in jobs: job.terminate()
        results = return_dict.values()
        res_len = len(results)
        merged = np.empty(shape=[len(points), 2])
        for i in range(0, res_len//2+1, 2):
            # merging points by y
            new_row = np.array(merge_by_y(results[i][0], results[i+1][0]))
            merged = np.append(merged, new_row)
        # merged = [item for sublist in merged for item in sublist]
        print(merged.shape)
        distances = [results[i][1][2] for i in range(res_len)]
        d = _boundary_merge(merged, distances, x_median)


        return merged, d


def closest_pair(points, max_workers):
    min_length = len(points) // max_workers
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    res = work(points, return_dict)
    print(res)


def medianSearch(list):
    return np.median(list)


def write_to_file(data, file_name):
    np.savetxt(file_name + ".csv", data, delimiter=",")


def plot_points(data, file_name, show=False):
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(file_name + ".png")
    if show:
        plt.show()


if __name__ == "__main__":
    np.random.seed(123)
    max_workers = 4
    size = 2 ** 4
    gen_data_margin = size
    input = np.random.randn(size, 2) * 100
    int_input = input.astype(int)
    write_to_file(int_input, "input")
    plot_points(int_input, "input")
    res = closest_pair(int_input, max_workers)
