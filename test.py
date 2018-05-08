def merge_by_y(left, right):
    """
    Implements merge of 2 sorted lists by the second coordinate
    :param args: support explicit left/right args, as well as a two-item
                tuple which works more cleanly with multiprocessing.
    :return: merged list
    """
    left_length, right_length = len(left), len(right)
    print(left_length)
    print(left.shape)
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


def sort_by_y(points):
    ind = np.argsort(points[:, 1])
    return points[ind]


if __name__ == "__main__":
    import numpy as np
    size = 4
    input1 = np.random.randn(size, 2) * 100
    print(input1)
    input2 = np.random.randn(size, 2) * 100
    print(input2)
    print(merge_by_y(sort_by_y(input1), sort_by_y(input2)))