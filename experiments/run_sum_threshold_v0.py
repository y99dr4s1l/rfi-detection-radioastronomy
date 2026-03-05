import numpy as np
import pandas as pd
from data.luserna_loader import load_luserna, load_luserna_truth
from preprocessing.spectrogram import polynomial_detrend, extract_and_split_patches
from methods.statistical.sum_threshold import chi_threshold5, sumthreshold
from evaluation.metrics import compute_metrics
from evaluation.timing import Timer
from visualization.plots import plot_detection_result

# --- CONFIG ---
DATA_PATH = 'data/raw/luserna'
NAME = 'PG01'
CHI0 = 2.4
MU0 = 0
I = 4
PATCH_SIZE = (512, 512)
TRAIN_SIZE = 492
MAX_PATCHES = 612
RANDOM_SEED = 42
RESULTS_PATH = 'experiments/results/sumthreshold_luserna.csv'

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
    max_patches=MAX_PATCHES,
    random_seed=RANDOM_SEED
)

# --- INFERENCE ---
chi = chi_threshold5(i=I, chi1=CHI0, target_value=MU0)
print(f'Running SumThreshold (chi0={CHI0}, i={I}) on {test_data.shape[0]} patches...')
print(f'Chi: {chi}')

pred_masks = np.zeros(test_masks.shape, dtype=bool)

with Timer() as t:
    for v in range(test_data.shape[0]):
        pred_masks[v, ..., 0] = sumthreshold(test_data[v, ..., 0], chi, output=False)

print(f'Inference time: {t}')

# --- METRICS ---
metrics = compute_metrics(test_masks, pred_masks)
metrics['time_seconds'] = t.elapsed
metrics['chi0'] = CHI0
metrics['i'] = I
metrics['dataset'] = 'luserna'
metrics['method'] = 'sumthreshold'

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
rfi_patches = np.where(np.any(test_masks > 0, axis=(1, 2, 3)))[0]
sample_idx = rfi_patches[0] if len(rfi_patches) > 0 else 0
print(f'Plotting patch {sample_idx} (RFI patches available: {len(rfi_patches)})')

plot_detection_result(
    spectrogram=test_data[sample_idx],
    ground_truth=test_masks[sample_idx],
    predicted_mask=pred_masks[sample_idx],
    title=f'SumThreshold — chi0={CHI0}, i={I}',
    save_path='experiments/results/sumthreshold_luserna_sample.png'
)