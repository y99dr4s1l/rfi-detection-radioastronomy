import numpy as np
import pandas as pd
import os

os.makedirs('data/raw/luserna', exist_ok=True)

rows, cols = 6144, 1024
patch_size = 512

np.random.seed(42)
data = np.random.randn(rows, cols).astype(np.float32)
noise_std = np.std(data)
rfi_intensity = 10.0 * noise_std

truth = np.zeros((rows, cols), dtype=bool)

# per ogni patch verticale (lungo rows) decidi con p=0.8 se inserire una barra
n_patches_y = rows // patch_size  # 12 patch
n_patches_x = cols // patch_size  # 2 patch

for py in range(n_patches_y):
    for px in range(n_patches_x):
        if np.random.rand() < 0.8:
            # coordinata y centrale della barra nella patch, lievemente diagonale
            y_start = py * patch_size
            x_start = px * patch_size
            x_end = x_start + patch_size

            for x in range(x_start, x_end):
                # pendenza: 1 pixel ogni 100 colonne
                x_local = x - x_start
                y_center = y_start + patch_size // 2 + int(x_local * patch_size / (100 * patch_size))
                for dy in range(-1, 2):  # banda 3 pixel
                    y = y_center + dy
                    if 0 <= y < rows:
                        data[y, x] += rfi_intensity
                        truth[y, x] = True

index = np.arange(rows, dtype=np.int64) * 1_000_000
pd.DataFrame(data, index=index).to_hdf('data/raw/luserna/PG01.h5', key='data')
pd.DataFrame(truth, index=index).to_hdf('data/raw/luserna/truth_flag.h5', key='bool_data')

n_rfi_patches = sum(
    1 for py in range(n_patches_y) for px in range(n_patches_x)
    if truth[py*patch_size:(py+1)*patch_size, px*patch_size:(px+1)*patch_size].any()
)
print(f'Mock data created: {rows}x{cols}')
print(f'RFI patches: {n_rfi_patches}/{n_patches_y*n_patches_x}')
print(f'RFI pixels: {truth.sum()}')