import numpy as np
import matplotlib.pyplot as plt


def plot_detection_result(
    spectrogram: np.ndarray,
    ground_truth: np.ndarray,
    predicted_mask: np.ndarray,
    title: str = None,
    save_path: str = None
) -> None:
    """
    Plots the spectrogram, ground truth mask, and predicted mask side by side.

    Args:
        spectrogram: 2D array of shape (time, frequency).
        ground_truth: 2D boolean array, same shape as spectrogram.
        predicted_mask: 2D boolean array, same shape as spectrogram.
        title: Optional title for the figure.
        save_path: If provided, saves the figure to this path instead of showing it.
    """
    if spectrogram.ndim == 3:
        spectrogram = spectrogram[..., 0]
    if ground_truth.ndim == 3:
        ground_truth = ground_truth[..., 0]
    if predicted_mask.ndim == 3:
        predicted_mask = predicted_mask[..., 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(spectrogram, aspect='auto', origin='lower')
    axes[0].set_title('Spectrogram')
    axes[0].set_xlabel('Frequency')
    axes[0].set_ylabel('Time')

    axes[1].imshow(ground_truth, aspect='auto', origin='lower', cmap='Reds')
    axes[1].set_title('Ground Truth')
    axes[1].set_xlabel('Frequency')

    axes[2].imshow(predicted_mask, aspect='auto', origin='lower', cmap='Reds')
    axes[2].set_title('Predicted Mask')
    axes[2].set_xlabel('Frequency')

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()