from typing import List
import numpy as np
from scipy.interpolate import CubicSpline


def np_mode(array: np.ndarray, exclude_nan: bool = True) -> any:
    """
    Find mode value in a 1D array

    Args:
        array: an array of any shape, but it will be treated as a 1D array
        exclude_nan: whether to exclude nan values when finding mode

    Returns:
        the mode value, if `exclude_nan` is True and the whole input array is NaN, return None
    """
    if exclude_nan:
        array = array[~np.isnan(array)]
        if len(array) == 0:
            return None
    val, count = np.unique(array, return_counts=True)
    mode_ = val[np.argmax(count)]
    return mode_


def gen_random_curves(length: int, num_curves: int, sigma=0.2, knot=4):
    """

    Args:
        length:
        num_curves:
        sigma:
        knot:

    Returns:
        array shape [length, num curves]
    """
    xx = np.arange(0, length, (length - 1) / (knot + 1))
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, num_curves))
    x_range = np.arange(length)
    curves = np.array([CubicSpline(xx, yy[:, i])(x_range) for i in range(num_curves)]).T
    return curves


def interval_intersection(intervals: List[List[List[int]]]) -> List[List[int]]:
    """
    Find the intersection of multiple interrupted timeseries, each of which contains multiple segments represented by a
    pair of start & end timestamp.

    Args:
        intervals: a 3-level list;
            1st level: each element represents a timeseries;
            2nd level: each element is a segment in a timeseries;
            3rd level: 2 elements are start & end timestamps of a segment

    Returns:
        The intersection is also a timeseries with the same format as one element in the input list.
    """
    if len(intervals) == 0:
        raise ValueError("The input list doesn't have any element.")
    if len(intervals) == 1:
        return intervals[0]

    # index indicating current position for each interval list
    pos_indices = np.zeros(len(intervals), dtype=int)
    # lengths of all ranges
    all_len = np.array([len(interval) for interval in intervals])

    result = []
    while np.all(pos_indices < all_len):
        # the startpoint of the intersection
        lo = max([intervals[interval_idx][pos_idx][0] for interval_idx, pos_idx in enumerate(pos_indices)])
        # the endpoint of the intersection
        endpoints = [intervals[interval_idx][pos_idx][1] for interval_idx, pos_idx in enumerate(pos_indices)]
        hi = min(endpoints)

        # save result if there is an intersection among segments
        if lo < hi:
            result.append([lo, hi])

        # remove the interval with the smallest endpoint
        pos_indices[np.argmin(endpoints)] += 1

    return result
