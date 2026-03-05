import numpy as np
import scipy.ndimage as im
from scipy.ndimage import convolve1d
from scipy import stats


def chi_threshold5(i, chi1, target_value, acc=0):
    """
    Computes a dictionary of thresholds with exponential decay.

    Args:
        i: Number of iterations (powers of 2 from 2^1 to 2^i).
        chi1: Initial threshold value.
        target_value: Target value for decay.
        acc: Accuracy parameter. Default is 0.

    Returns:
        dict: Keys are window sizes (powers of 2), values are thresholds.
    """
    chi = {}
    decay_factor = 1 / np.sqrt(2)
    for iteration in range(1, i + 1):
        chi[2 ** iteration] = (
            (chi1 - target_value) * ((decay_factor * (1 - acc)) ** np.log2(2 ** iteration + 1))
        )
    return chi


def calculate_thresholds(iterations: int, chi0: float, exp_base: float = 1.5) -> dict:
    """
    Computes thresholds following Offringa's formulation.

    Args:
        iterations: Number of iterations (window sizes as powers of 2).
        chi0: Base threshold value.
        exp_base: Exponential base for scaling. Default is 1.5.

    Returns:
        dict: Keys are window sizes, values are thresholds.
    """
    lengths = [2 ** i for i in range(iterations)]
    thresholds = {}
    for length in lengths:
        thresholds[length] = chi0 * (exp_base ** np.log2(max(length, 1))) / max(length, 1)
    return thresholds


def winsorized_mode(data: np.ndarray, limits: float = 0.05) -> float:
    """
    Computes the mode of winsorized data.

    Args:
        data: Input 2D array.
        limits: Fraction to winsorize on each side. Default is 0.05.

    Returns:
        float: Mode of the winsorized data.
    """
    winsorized_data = stats.mstats.winsorize(data, limits=limits)
    mode_result = stats.mode(winsorized_data.compressed(), keepdims=True)
    return mode_result[0][0]


def sum_threshold_horizontal_optimized(
    input_arr: np.ndarray,
    mask: np.ndarray,
    length: int,
    threshold: float
) -> np.ndarray:
    """
    Applies horizontal SumThreshold flagging for a single window length.

    Args:
        input_arr: 2D input spectrogram array.
        mask: Current boolean mask.
        length: Window length.
        threshold: Threshold value for this window.

    Returns:
        np.ndarray: Updated boolean mask.
    """
    height, width = input_arr.shape
    if length > width:
        return mask

    masked_input = np.where(mask, 0, input_arr)
    masked_count = np.where(mask, 0, 1)
    kernel = np.ones(length)

    sums = convolve1d(masked_input, kernel, axis=1, mode='constant', origin=-(length // 2))
    counts = convolve1d(masked_count, kernel, axis=1, mode='constant', origin=-(length // 2))

    valid_sums = sums[:, length - 1:]
    valid_counts = counts[:, length - 1:]

    with np.errstate(divide='ignore', invalid='ignore'):
        averages = np.abs(valid_sums / valid_counts)

    threshold_mask = (averages > threshold) & (valid_counts > 0)

    for x_offset in range(length):
        mask[:, x_offset:x_offset + threshold_mask.shape[1]] |= threshold_mask

    return mask


def sumthreshold_optimized(input_arr: np.ndarray, chi_dict: dict) -> np.ndarray:
    """
    Applies SumThreshold flagging iteratively over increasing window sizes.

    Args:
        input_arr: 2D input spectrogram array.
        chi_dict: Dictionary of thresholds keyed by window size.

    Returns:
        np.ndarray: Boolean mask of flagged pixels.
    """
    mask = np.zeros_like(input_arr, dtype=bool)
    for length in sorted(chi_dict.keys()):
        threshold = chi_dict[length]
        mask = sum_threshold_horizontal_optimized(input_arr, mask, length, threshold)
    return mask


def sumthreshold(arr: np.ndarray, chi: dict, output: bool = False):
    """
    Original SumThreshold implementation.

    Args:
        arr: 2D input array.
        chi: Dictionary of thresholds keyed by window size.
        output: If True, returns (mask, S, chi). Default is False.

    Returns:
        np.ndarray or tuple: Boolean mask, or (mask, S, chi) if output=True.
    """
    mask = np.zeros(shape=arr.shape, dtype=bool)
    S = {}
    for key in chi.keys():
        w = key
        t = chi[w] * w
        new_arr = arr.copy()
        flag = mask.copy()
        new_arr[mask] = chi[w]
        kernel_m = np.ones((1, w))
        sum_arr = im.convolve(new_arr, kernel_m, mode='constant', cval=0, origin=0)
        sum_arr = sum_arr[:, int(w / 2) - 1:-int(w / 2)]
        S[w] = sum_arr
        flag[:, int(w / 2) - 1:-int(w / 2)][sum_arr > t] = True
        for i in range(1, w):
            flag = flag + np.roll(flag, 1, axis=1)
        flag = flag > 0
        mask = flag + mask

    if output:
        return mask, S, chi
    return mask