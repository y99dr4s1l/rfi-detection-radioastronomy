import numpy as np
import pandas as pd
import time
from data.luserna_loader import load_luserna, load_luserna_truth
from preprocessing.spectrogram import polynomial_detrend, extract_and_split_patches
from methods.statistical.cusum import CUSUM
from evaluation.metrics import compute_metrics
from evaluation.timing import Timer
from visualization.plots import plot_detection_result

# --- CONFIG ---
DATA_PATH = 'data/raw/luserna'
NAME = 'PG01'
K = 0.46
H = 1.1
PATCH_SIZE = (512, 512)
TRAIN_SIZE = 492
RANDOM_SEED = 42
RESULTS_PATH = 'experiments/results/cusum_luserna.csv'

# --- LOAD ---
print('Loading data...')
data = load_luserna(NAME, path=DATA_PATH, powers=False)
truth = load_luserna_truth(path=DATA_PATH)

data_np = data.to_numpy().astype(np.float32)
truth_np = truth.to_numpy().astype(bool)

# --- PREPROCESSING ---
print('Preprocessing...')
data_np = polynomial_detrend(data_np, degree=2)
data_np = np.clip(data_np, -10, 10)

_, _, test_data, test_masks = extract_and_split_patches(
    data_np, truth_np,
    patch_size=PATCH_SIZE,
    train_size=TRAIN_SIZE,
    max_patches=612,
    random_seed=RANDOM_SEED
)

# --- INFERENCE ---
print(f'Running CUSUM (k={K}, h={H}) on {test_data.shape[0]} patches...')
pred_masks = np.zeros(test_masks.shape, dtype=bool)

with Timer() as t:
    for v in range(test_data.shape[0]):
        pred_masks[v, ..., 0] = CUSUM(test_data[v, ..., 0], k=K, h=H, output=False)

print(f'Inference time: {t}')

# --- METRICS ---
metrics = compute_metrics(test_masks, pred_masks)
metrics['time_seconds'] = t.elapsed
metrics['k'] = K
metrics['h'] = H
metrics['dataset'] = 'luserna'
metrics['method'] = 'cusum'

print(f"\nResults:")
print(f"  Precision : {metrics['precision']:.4f}")
print(f"  Recall    : {metrics['recall']:.4f}")
print(f"  F1        : {metrics['f1']:.4f}")
print(f"  Time      : {metrics['time_seconds']:.2f}s")

# --- SAVE RESULTS ---
df = pd.DataFrame([metrics])
df.to_csv(RESULTS_PATH, index=False)
print(f'\nResults saved to {RESULTS_PATH}')

# --- PLOT ---
sample_idx = 0
plot_detection_result(
    spectrogram=test_data[sample_idx],
    ground_truth=test_masks[sample_idx],
    predicted_mask=pred_masks[sample_idx],
    title=f'CUSUM — k={K}, h={H}',
    save_path='experiments/results/cusum_luserna_sample.png'
)