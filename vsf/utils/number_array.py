from typing import List
import bezier
import numpy as np
from scipy.interpolate import CubicSpline


def interp_resample(arr: np.ndarray, old_freq: float, new_freq: float) -> np.ndarray:
    """
    Resample by linear interpolation for every channel in the data array.

    Args:
        arr: data array, shape [length, channel]
        old_freq: old sampling rate
        new_freq: new sampling rate

    Returns:
        new data array, shape [new length, channel]
    """
    assert len(arr.shape) == 2, 'Only support 2D array'

    data_duration = len(arr) / old_freq
    new_arr = []
    for i in range(arr.shape[1]):
        old_ts = np.arange(0, data_duration, 1 / old_freq)
        new_ts = np.arange(0, data_duration, 1 / new_freq)
        new_channel = np.interp(x=new_ts, xp=old_ts, fp=arr[:, i])
        new_arr.append(new_channel)
    new_arr = np.stack(new_arr, axis=-1)
    return new_arr


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


def gen_random_curves(length: int, num_curves: int, sigma=0.2, knot=4, method: str = 'bezier',
                      randomizer: np.random._generator.Generator = None):
    """
    Generate random curves

    Args:
        length: length of the curve(s) to be generated
        num_curves: number of curves to be generated
        sigma: warping magnitude (std)
        knot: number of turns in the curve(s)
        method: method to connect random points to form a random curve;
            currently support [cubic|beizer]; default: bezier
        randomizer: numpy random generator (with seed)

    Returns:
        array shape [length, num curves]
    """
    xx = np.arange(0, length, (length - 1) / (knot + 1))
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, num_curves)) if randomizer is None \
        else randomizer.normal(loc=1.0, scale=sigma, size=(knot + 2, num_curves))

    if method == 'bezier':
        x_range = np.linspace(0, 1, num=length, endpoint=True)
        curves = [bezier.Curve.from_nodes(np.array([xx, yy[:, i]])).evaluate_multi(x_range)[1]
                  for i in range(num_curves)]
    elif method == 'cubic':
        x_range = np.arange(length)
        curves = [CubicSpline(xx, yy[:, i])(x_range) for i in range(num_curves)]
    else:
        raise ValueError(f'Available methods: [cubic|beizer], but received: {method}')

    curves = np.array(curves).T
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    _ = np.asfortranarray([
        [0.0, 1.0],
        [0.0, 2.0],
        [0.0, 3.0],
    ])

    randomizer = np.random.default_rng(seed=2)
    cubic_curves = gen_random_curves(length=100, num_curves=2, sigma=0.2, knot=2, method='cubic', randomizer=randomizer)

    randomizer = np.random.default_rng(seed=2)
    bezier_curves = gen_random_curves(length=100, num_curves=2, sigma=0.2, knot=2, method='bezier',
                                      randomizer=randomizer)

    plt.subplot(1, 2, 1)
    plt.plot(cubic_curves[:, 0], label='cubic')
    plt.plot(bezier_curves[:, 0], label='bezier')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(cubic_curves[:, 1], label='cubic')
    plt.plot(bezier_curves[:, 1], label='bezier')
    plt.legend()

    plt.show()
