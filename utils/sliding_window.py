import numpy as np


def sliding_window(data: np.ndarray, window_size: int, step_size: int, get_last: bool = True) -> np.ndarray:
    """
    Sliding window along the first axis of the input array.
    Args:
        data:
        window_size:
        step_size:
        get_last:

    Returns:

    """
    num_windows = (len(data) - window_size) / step_size + 1
    if num_windows < 1:
        return np.empty([0, window_size, *data.shape[1:]], dtype=data.dtype)

    # if possible, run fast sliding window
    if window_size % step_size == 0:
        result = np.empty([int(num_windows), window_size, *data.shape[1:]], dtype=data.dtype)
        div = int(window_size / step_size)
        for window_idx, data_idx in enumerate(range(0, window_size, step_size)):
            new_window_data = data[data_idx:data_idx + (len(data) - data_idx) // window_size * window_size].reshape(
                [-1, window_size, *data.shape[1:]])

            new_window_idx = list(range(window_idx, int(num_windows), div))
            result[new_window_idx] = new_window_data
    # otherwise, run a regular loop
    else:
        result = np.array([data[i:i + window_size] for i in range(0, len(data) - window_size + 1, step_size)])

    if get_last and (num_windows % 1 != 0):
        result = np.concatenate([result, [data[-window_size:]]])

    return result


def shifting_window(data: np.ndarray, window_size: int, max_num_windows: int, min_step_size: int,
                    start_idx: int, end_idx: int) -> np.ndarray:
    """
    Get window(s) from an array while ensuring a certain part of the array is included.

    Args:
        data: array shape [data length, ...]
        window_size: window size
        max_num_windows: desired number of windows, the actual number returned maybe smaller,
            depending on `min_step_size`
        min_step_size: only get multiple windows if 2 windows are farther than `min_step_size` from each other
        start_idx: start index of the required part (inclusive)
        end_idx: end index of the required part (inclusive)

    Returns:
        array shape [num window, window size, ...]
    """
    end_idx = end_idx + 1

    # exception 1: data array is too small compared to window size
    if len(data) < window_size:
        return np.empty([0, window_size, *data.shape[1:]], dtype=data.dtype)
    elif len(data) == window_size:
        return np.expand_dims(data, axis=0)

    # exception 2: required part is not entirely in data array
    end_idx = min(end_idx, len(data))
    start_idx = max(start_idx, 0)

    # data array is large enough to contain both required part and window size
    first_window_start_idx = max(min(end_idx - window_size, start_idx), 0)
    last_window_start_idx = min(max(end_idx - window_size, start_idx), len(data) - window_size)
    assert last_window_start_idx >= first_window_start_idx

    # if there's only 1 window
    if first_window_start_idx == last_window_start_idx:
        windows = np.expand_dims(data[first_window_start_idx:first_window_start_idx + window_size], axis=0)
        return windows

    # otherwise, get windows by linspace
    max_num_windows = min(max_num_windows,
                          np.ceil((last_window_start_idx - first_window_start_idx + 1) / min_step_size).astype(int))
    window_start_indices = np.linspace(first_window_start_idx, last_window_start_idx, num=max_num_windows,
                                       endpoint=True, dtype=int)

    windows = np.array([data[i:i + window_size] for i in window_start_indices])
    return windows


if __name__ == '__main__':
    result = shifting_window(
        data=np.arange(1000),
        window_size=10,
        max_num_windows=3,
        min_step_size=5,
        start_idx=1,
        end_idx=5
    )

    print(result.shape)
    print(result)
