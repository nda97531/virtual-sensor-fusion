from typing import Union, List

import numpy as np
from loguru import logger
from transforms3d.axangles import axangle2mat

from my_py_utils.my_py_utils.number_array import gen_random_curves


def format_range(x: any, start_0: bool) -> np.ndarray:
    """
    Turn an arbitrary input into a range. For example:
        1 to [-1, 1] or [0, 1]
        None to [0, 0]
        [-1, 1] will be kept intact

    Args:
        x: any input
        start_0: if input x is a scalar and this is True, the starting point of the range will be 0,
            otherwise it will be -abs(x)

    Returns:
        a np array of 2 element [range start, range end]
    """
    if x is None:
        x = [0, 0]
    if isinstance(x, float) or isinstance(x, int):
        x = abs(x)
        x = [0 if start_0 else -x, x]
    assert len(x) == 2, 'x must be a scalar or a list/tuple of 2 numbers'
    return np.array(x)


class Augmenter:
    # whether the augmenter's randomizer depend on data shape
    DATA_DEPENDENT_RANDOM: bool

    def __init__(self, p: float = 1, random_seed: Union[int, None] = None):
        """
        Abstract base class for augmenters

        Args:
            p: range (0, 1], possibility to apply this augmenter each time `run` is called
            random_seed: random seed for this Augmenter; if None, no fixed seed is used
        """
        if p <= 0 or p > 1:
            logger.info(f'Augment probability is not in [0,1): {p}')
        self.p = p
        self.randomizer = np.random.default_rng(seed=random_seed)

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        """
        Apply the augmentation

        Args:
            org_data: a single sample (without batch dim)

        Returns:
            numpy array of the same shape
        """
        raise NotImplementedError()

    def _copy_data(self, org_data: np.ndarray) -> np.ndarray:
        """
        Return a copy of the original data array to avoid modifying the raw dataset.
        """
        return org_data.copy()

    def run(self, org_data: np.ndarray) -> np.ndarray:
        """
        The main method to apply augmentation on raw data

        Args:
            org_data: numpy array [time step, channel]

        Returns:
            augmented data
        """
        if (self.p >= 1) or (self.randomizer.random() < self.p):
            data = self._copy_data(org_data)
            data = self._apply_logic(data)
            return data

        return org_data


class ComposeAugmenters(Augmenter):
    DATA_DEPENDENT_RANDOM = False

    def __init__(self, augmenters: List[Augmenter], shuffle_augmenters: bool = True,
                 p: float = 1, random_seed: Union[int, None] = None):
        """
        Combine many augmenters into a single one

        Args:
            augmenters: list of Augmenter objects
            shuffle_augmenters: whether to shuffle the order of augmenters when applying
            p: range (0, 1], possibility to apply this augmenter each time `run` is called
            random_seed: random seed for this Augmenter
        """
        super().__init__(p=p, random_seed=random_seed)
        self.augmenters = augmenters
        self.shuffle_augmenters = shuffle_augmenters if len(augmenters) > 1 else False

    def _copy_data(self, org_data: np.ndarray) -> np.ndarray:
        """
        No copy needed because all elements augmenters do this
        """
        return org_data

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        for i in self.randomizer.permutation(range(len(self.augmenters))) if self.shuffle_augmenters \
                else range(len(self.augmenters)):
            org_data = self.augmenters[i].run(org_data)
        return org_data


class Rotation3D(Augmenter):
    DATA_DEPENDENT_RANDOM = True

    def __init__(self, angle_range: Union[list, tuple, float] = 180, rot_axis: np.ndarray = None,
                 separate_triaxial: bool = False, p: float = 1, random_seed: Union[int, None] = None) -> None:
        """
        Rotate tri-axial data in a random axis. This is for tri-axial inertial data.

        Args:
            angle_range: (degree) the angle is randomised within this range;
                if this is a list, randomly pick an angle in this range;
                if it's a float, the range is [-float, float]
            rot_axis: rotation axis, an array with 3 element (x, y, z); default: random every run
            separate_triaxial: if True, use different random angles for each of the tri-axial sensor. For example, array
                shape [n, 6] is 2 tri-axial sensors, each has 3 features.
            p: probability to apply this augmenter each time it is called
            random_seed: random seed for this Augmenter
        """
        super().__init__(p=p, random_seed=random_seed)

        self.angle_range = format_range(angle_range, start_0=False) / 180 * np.pi
        self.separate_triaxial = separate_triaxial
        self.rot_axis = rot_axis

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        assert (len(org_data.shape) == 2) and (org_data.shape[1] % 3 == 0), \
            f"expected data shape: [any length, channel%3==0], got {org_data.shape}"

        if self.separate_triaxial:
            num_tri = org_data.shape[1] // 3
            angles = self.randomizer.uniform(low=self.angle_range[0], high=self.angle_range[1], size=num_tri)
            direction_vectors = self.randomizer.uniform(-1, 1, size=[num_tri, 3]) \
                if self.rot_axis is None else np.stack([self.rot_axis] * num_tri)

            # transpose data to shape [channel, time step]
            data = org_data.T

            # for every 3 channels
            for ith, i in enumerate(range(0, data.shape[0], 3)):
                data[i:i + 3] = self.rotate(data[i:i + 3], angles[ith], direction_vectors[ith])
        else:
            angle = self.randomizer.uniform(low=self.angle_range[0], high=self.angle_range[1])
            direction_vector = self.randomizer.uniform(-1, 1, size=3) \
                if self.rot_axis is None else self.rot_axis

            # transpose data to shape [channel, time step]
            data = org_data.T

            # for every 3 channels
            for i in range(0, data.shape[0], 3):
                data[i:i + 3] = self.rotate(data[i:i + 3], angle, direction_vector)

        # transpose back to [time step, channel]
        data = data.T
        return data

    @staticmethod
    def rotate(data: np.ndarray, angle: float, axis: np.ndarray):
        """
        Rotate data array

        Args:
            data: data array, shape [3, n]
            angle: a random angle in radian
            axis: a 3-d vector, the axis to rotate around

        Returns:
            rotated data of the same format as the input
        """
        rot_mat = axangle2mat(axis, angle)
        data = np.matmul(rot_mat, data)
        return data


class HorizontalFlip(Augmenter):
    DATA_DEPENDENT_RANDOM = False

    def __init__(self, p=0.5, random_seed: Union[int, None] = None):
        """
        Flip skeleton data horizontally by multiplying x-axis with -1.
        Please make sure data is normalised (around 0).
        """
        assert p < 1, 'Do not always apply flip!'
        super().__init__(p=p, random_seed=random_seed)

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        """
        Flip data horizontally (flip x-axis)

        Args:
            org_data: array shape [time step, num joints * 2(x,y)]

        Returns:
            same shape array
        """
        org_data[:, np.arange(0, org_data.shape[1], 2)] *= -1
        return org_data


class TimeWarp(Augmenter):
    DATA_DEPENDENT_RANDOM = False

    def __init__(self, sigma: float = 0.2, knot: int = 4, p: float = 1, random_seed: Union[int, None] = None):
        """
        Time warping augmentation

        Args:
            sigma: warping magnitude (std)
            knot: number of knot to generate a random curve to distort timestamps
            p: probability to apply this augmenter each time it is called
            random_seed: random seed for this Augmenter
        """
        super().__init__(p=p, random_seed=random_seed)
        self.sigma = sigma
        self.knot = knot

    def distort_time_steps(self, length: int, num_curves: int = 1):
        """
        Create distort timestamps for warping

        Args:
            length: length of the time-series (number of timestamps)
            num_curves: number of arrays to generate
        Returns:
            numpy array shape [length, num_curves]
        """
        tt = gen_random_curves(length, num_curves, self.sigma, self.knot, randomizer=self.randomizer)
        tt_cum = np.cumsum(tt, axis=0)

        # Make the last value equal length
        tt_cum = (tt_cum - tt_cum[0]) / (tt_cum[-1] - tt_cum[0]) * (length - 1)
        return tt_cum

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        # create new timestamp for all channels
        tt_new = self.distort_time_steps(org_data.shape[-2]).squeeze()
        x_range = np.arange(org_data.shape[0])
        data = np.array([
            np.interp(x_range, tt_new, org_data[:, i]) for i in range(org_data.shape[-1])
        ]).T
        return data


class Scale(Augmenter):
    DATA_DEPENDENT_RANDOM = True

    def __init__(self, sigma: float = 0.1, p: float = 1, random_seed: Union[int, None] = None):
        """
        Multiply the whole data with a scalar

        Args:
            sigma: std for randomly generated scale factor
        """
        super().__init__(p, random_seed)
        assert sigma >= 0, 'sigma must be non negative'
        self.sigma = sigma

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        # shape [1, channel]
        scale = self.randomizer.normal(loc=1, scale=self.sigma, size=[1, org_data.shape[1]])
        org_data *= scale
        return org_data


class Jittering(Augmenter):
    DATA_DEPENDENT_RANDOM = True

    def __init__(self, sigma: float, p: float = 1, random_seed: Union[int, None] = None):
        """
        Add Gaussian noise to data

        Args:
            sigma: standard deviation of noise
        """
        super().__init__(p, random_seed)
        self.sigma = sigma

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        noise = self.randomizer.normal(loc=0, scale=self.sigma, size=org_data.shape)
        org_data += noise
        return org_data


class MagnitudeWarp(Augmenter):
    DATA_DEPENDENT_RANDOM = True

    def __init__(self, sigma: float = 0.2, knot_range: Union[int, list] = 4, p: float = 1,
                 random_seed: Union[int, None] = None):
        """

        Args:
            sigma: warping magnitude (std)
            knot_range: number of knot to generate a random curve to distort timestamps
        """
        super().__init__(p, random_seed)
        self.sigma = sigma
        self.knot_range = format_range(knot_range, start_0=True)
        # add one here because upper bound is exclusive when randomising
        self.knot_range[1] += 1

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        n_knot = self.randomizer.integers(self.knot_range[0], self.knot_range[1])
        curves = gen_random_curves(len(org_data), num_curves=org_data.shape[1], sigma=self.sigma, knot=n_knot,
                                   randomizer=self.randomizer)
        org_data *= curves
        return org_data


class Permutation(Augmenter):

    def __init__(self, num_parts: int = 4, min_part_weight: float = 0.1,
                 p: float = 1, random_seed: Union[int, None] = None):
        """
        Randomly split data into many parts and shuffle them

        Args:
            num_parts: number of parts to split
            min_part_weight: minimum length weight of each part;
                example: weight=0.2, data length=100 => min part length = 100 * 0.2 = 20
        """
        super().__init__(p, random_seed)
        assert num_parts * min_part_weight <= 1, \
            (f'Can\'t divide array into {num_parts} parts '
             f'with a minimum length for each part of {min_part_weight * 100}%')

        self.num_parts = num_parts
        self.min_part_weight = min_part_weight

    def _apply_logic(self, org_data: np.ndarray) -> np.ndarray:
        # calculate random lengths for N parts
        max_part_weight = (1 - self.min_part_weight) / (self.num_parts - 1)
        part_weights = self.randomizer.uniform(self.min_part_weight, max_part_weight, size=self.num_parts)
        part_weights = part_weights / part_weights.sum() * len(org_data)
        part_lengths = part_weights.astype(int)
        part_lengths[-1] = len(org_data) - part_lengths[:-1].sum()
        part_idx = np.cumsum(part_lengths)[:-1]

        # split data using calculated part lengths
        parts = np.split(org_data, part_idx)

        # randomly shuffle parts
        self.randomizer.shuffle(parts)
        data = np.concatenate(parts)
        return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data1 = np.arange(0, 10, 0.1).reshape([-1, 1])
    data2 = np.array([
        np.arange(10, 20, 0.1),
        np.arange(10, 20, 0.1) + 1
    ]).T

    new_data1 = Permutation(num_parts=4, min_part_weight=0.1, random_seed=1).run(data1)
    new_data2 = Permutation(num_parts=4, min_part_weight=0.1, random_seed=1).run(data2)

    plt.plot(data1, label='org1')
    plt.plot(data2, label='org2')
    plt.plot(new_data1, label='aug1')
    plt.plot(new_data2, label='aug2')

    plt.legend()
    plt.grid()
    plt.show()
