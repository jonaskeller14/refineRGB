import numpy as np


def n_unique_per_row(array: np.ndarray, return_row_counts_equals_2 = False):
    """
    Computes the number of unique elements of 2D-array per row.
    :param array: 2D np.ndarray
    :param return_row_counts_equals_2:
    :return: 1D np.ndarray with numbers of unique elements per row
    """
    array_sorted = np.sort(array, axis=1)
    n_unique = (array_sorted[:, 1:] != array_sorted[:, :-1]).sum(axis=1) + 1
    if return_row_counts_equals_2:
        even = array_sorted[:, 0::2]
        odd = array_sorted[:, 1::2]
        trans1 = array_sorted[:, 1:-1:2]  # checks that each entry occurs not 4,6,8 times
        trans2 = array_sorted[:, 2:-1:2]
        counts_bol = np.all(even == odd, axis=1) * np.all(trans1 != trans2, axis=1)
        return n_unique, counts_bol
    else:
        return n_unique



