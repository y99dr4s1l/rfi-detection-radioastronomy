import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA


def _cusum_scalar(arr: np.ndarray, k: float = 0.21) -> float:
    """
    Computes CUSUM statistic as a scalar feature for a single patch.

    Args:
        arr: 2D input array.
        k: Allowance parameter. Default is 0.21.

    Returns:
        float: Maximum CUSUM value.
    """
    C = np.zeros(shape=arr.shape)
    for t in range(1, arr.shape[1]):
        c = C[:, t - 1] + arr[:, t] - k
        c[c < 0] = 0
        C[:, t] = c
    return np.max(C)


def extract_features_from_images(
    images: np.ndarray,
    k_cusum: float = 0.21,
    significant_pca_indices: list = [0, 2, 3, 4, 5]
) -> np.ndarray:
    """
    Extracts statistical and PCA features from a set of small patches.

    Features extracted per patch:
        - Median
        - Energy (sum of squares)
        - Normalized variance (var / mean^2)
        - CUSUM scalar
        - GLCM contrast
        - GLCM dissimilarity
        - Selected PCA components

    Args:
        images: Array of shape (n, h, w, 1). Patches must be square
                with h in {4, 8, 16}.
        k_cusum: Allowance parameter for CUSUM scalar feature. Default is 0.21.
        significant_pca_indices: Indices of PCA components to include.

    Returns:
        np.ndarray: Feature matrix of shape (n, n_features).
    """
    try:
        from skimage.feature import graycomatrix as greycomatrix
        from skimage.feature import graycoprops as greycoprops
    except ImportError:
        try:
            from skimage.feature.texture import greycomatrix, greycoprops
        except ImportError:
            print('WARNING: GLCM features disabled. Install scikit-image >= 0.19.')
            greycomatrix = greycoprops = None

    if images.ndim == 3:
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)

    n, h, w, c = images.shape
    assert h == w and h in {4, 8, 16}, \
        f'Patches must be square with size in {{4, 8, 16}}, got {h}x{w}'

    n_stat_features = 6
    n_pca_features = len(significant_pca_indices)
    total_features = n_stat_features + n_pca_features
    features_matrix = np.zeros((n, total_features))

    flattened = images.reshape(n, h * w)
    pca = PCA(n_components=max(significant_pca_indices) + 1)
    pca_features = pca.fit_transform(flattened)

    for i in tqdm(range(n), desc='Feature extraction'):
        img = images[i, :, :, 0]
        mean_val = np.mean(img)
        var_val = np.var(img)
        idx = 0

        # 1. Median
        features_matrix[i, idx] = np.median(img)
        idx += 1

        # 2. Energy
        features_matrix[i, idx] = np.sum(img ** 2)
        idx += 1

        # 3. Normalized variance
        features_matrix[i, idx] = var_val / (mean_val ** 2) if abs(mean_val) > 1e-10 else 1e6
        idx += 1

        # 4. CUSUM scalar
        features_matrix[i, idx] = _cusum_scalar(img, k=k_cusum)
        idx += 1

        # 5-6. GLCM contrast and dissimilarity
        if greycomatrix is not None:
            try:
                img_norm = np.round(
                    (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10) * 255
                ).astype(np.uint8)
                n_levels = min(16, len(np.unique(img_norm)))
                if np.max(img_norm) >= n_levels:
                    img_norm = np.floor(
                        img_norm * (n_levels - 1) / np.max(img_norm)
                    ).astype(np.uint8)
                glcm = greycomatrix(
                    img_norm,
                    distances=[1],
                    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                    levels=n_levels,
                    symmetric=True,
                    normed=True
                )
                features_matrix[i, idx] = greycoprops(glcm, 'contrast').mean()
                idx += 1
                features_matrix[i, idx] = greycoprops(glcm, 'dissimilarity').mean()
                idx += 1
            except Exception:
                features_matrix[i, idx] = 0
                idx += 1
                features_matrix[i, idx] = 0
                idx += 1
        else:
            idx += 2

        # 7. PCA components
        for j, pca_idx in enumerate(significant_pca_indices):
            features_matrix[i, idx + j] = pca_features[i, pca_idx]

    return features_matrix


def transform_bool_labels(labels: np.ndarray) -> np.ndarray:
    """
    Converts 4D boolean mask patches to binary patch-level labels.
    A patch is labeled 1 if it contains at least one flagged pixel.

    Args:
        labels: Boolean array of shape (n, h, w, 1).

    Returns:
        np.ndarray: Binary label array of shape (n,).
    """
    if labels.ndim != 4:
        raise ValueError(f'Labels must have 4 dimensions, got shape {labels.shape}')

    binary_labels = np.zeros(labels.shape[0], dtype=int)
    for i in range(labels.shape[0]):
        if np.any(labels[i]):
            binary_labels[i] = 1

    print(f'Label distribution: {dict(zip(*np.unique(binary_labels, return_counts=True)))}')
    return binary_labels


def prepare_features(
    images: np.ndarray,
    labels: np.ndarray,
    k_cusum: float = 0.21
) -> tuple:
    """
    Extracts features and transforms labels for ML training.

    Args:
        images: Array of shape (n, h, w, 1).
        labels: Boolean array of shape (n, h, w, 1).
        k_cusum: Allowance parameter for CUSUM scalar feature.

    Returns:
        Tuple of (X, y) where X is the feature matrix and y the binary labels.
    """
    X = extract_features_from_images(images, k_cusum=k_cusum)
    y = transform_bool_labels(labels)
    return X, y


def create_features_dataframe(features_matrix: np.ndarray, labels=None):
    """
    Creates a pandas DataFrame from the feature matrix with column names.

    Args:
        features_matrix: Feature matrix of shape (n, n_features).
        labels: Optional binary label array of shape (n,).

    Returns:
        pd.DataFrame: Feature DataFrame with named columns.
    """
    import pandas as pd

    stat_columns = [
        'median',
        'energy',
        'normalized_variance',
        'cusum',
        'glcm_contrast',
        'glcm_dissimilarity',
    ]
    n_pca = features_matrix.shape[1] - len(stat_columns)
    pca_columns = [f'pca_{i + 1}' for i in range(n_pca)]
    all_columns = stat_columns + pca_columns

    if len(all_columns) != features_matrix.shape[1]:
        missing = features_matrix.shape[1] - len(all_columns)
        all_columns += [f'feature_{i + len(all_columns) + 1}' for i in range(missing)]

    df = pd.DataFrame(features_matrix, columns=all_columns)
    if labels is not None:
        df['label'] = labels

    return df


def reconstruct_from_patches(
    y_pred: np.ndarray,
    batch_size: int,
    img_size: int = 512,
    patch_size: int = 8
) -> np.ndarray:
    """
    Reconstructs full images from patch-level binary predictions.

    Args:
        y_pred: Binary prediction array of shape (n_patches,).
        batch_size: Number of original images.
        img_size: Size of the original image (assumed square). Default is 512.
        patch_size: Size of each patch (assumed square). Default is 8.

    Returns:
        np.ndarray: Reconstructed mask of shape (batch_size, img_size, img_size, 1).
    """
    patches_per_side = img_size // patch_size
    n_patches = batch_size * patches_per_side * patches_per_side

    y_recon = y_pred[:n_patches, np.newaxis, np.newaxis, np.newaxis]
    y_recon = np.broadcast_to(y_recon, (n_patches, patch_size, patch_size, 1)).copy()
    y_recon = y_recon.reshape(
        batch_size, patches_per_side, patches_per_side, patch_size, patch_size, 1
    ).transpose(0, 1, 3, 2, 4, 5).reshape(batch_size, img_size, img_size, 1)

    return y_recon