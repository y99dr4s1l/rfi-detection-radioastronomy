# RFI Detection in Radio Astronomy

Systematic comparison of traditional statistical methods and machine learning algorithms for automatic Radio Frequency Interference (RFI) detection in radio astronomical spectrograms.

## Overview

This work evaluates two deterministic methods (CUSUM and SumThreshold) and four machine learning algorithms (k-Nearest Neighbors, Random Forest, U-Net, and R-Net) for RFI identification in spectrograms acquired at 1.420 MHz (neutral hydrogen spectral line).

The analysis is conducted on two datasets:

- **Luserna**: real observations targeting Kepler catalog exoplanets, acquired at the Luserna San Giovanni astronomical observatory
- **LOFAR**: public benchmark dataset used for external validation

Performance is quantified using precision, recall, and F1-score, with computational timing analysis for each approach.

## Pipeline
```
Data Collection  - - - - - - - - - - - - - - -  Data acquisition, A/D conversion,
                      (RT)                                         spectrogram generation, CSV export
                        │
                        V
                   Annotation  - - - - - - - - - - - - - - - - -  Manual pixel-level labeling
                        │
                        V                                          Time de-trending,
                  Preprocessing  - - - - - - - - - - - - - - - -  standardization,
                        │                                          clipping, patching
       ┌────────────────┼────────────────┐
       │                │                │
 ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ - - - - - - -  Method selection
 │    CUSUM    │  │    K-NN     │  │   U-Net     │
 │SumThreshold │  │     RF      │  │   R-Net     │
 └─────────────┘  └─────────────┘  └─────────────┘
       │                │                │
       │                V                │                          Feature engineering
       │        Feature Extraction  - - -│ - - - - - - - - - - - -  and selection
       │                │                │
       │                └────────┬───────┘
       │                         │
       │                         V                                  Training, optimization,
       │                    Training  - - - - - - - - - - - - - - - validation
       │                         │
       V                         V
Threshold application       Prediction  - - - - - - - - - - - - -  Performance measurement
  on test data             on test data                             on test data
       │                         │
       └────────────┬────────────┘
                    │
                    V                                               Quantitative comparison,
               Evaluation  - - - - - - - - - - - - - - - - - - - - operational recommendations

```

## Project Structure
```
rfi-detection-radioastronomy/
├── data/
│   ├── raw/
│   │   ├── luserna/        # raw HDF5 spectrogram + ground truth
│   │   └── lofar/          # raw PKL dataset
│   ├── processed/          # preprocessed spectrograms
│   └── annotations/        # manual RFI labels
├── src/
│   ├── loaders/            # dataset loaders
│   ├── preprocessing/      # spectrogram normalization and preparation
│   ├── methods/
│   │   ├── statistical/    # CUSUM, SumThreshold
│   │   └── ml/             # KNN, Random Forest, U-Net, R-Net
│   ├── evaluation/         # metrics and timing benchmarks
│   └── visualization/      # plotting utilities
├── experiments/
│   ├── configs/            # YAML hyperparameter configurations
│   └── results/            # output CSV results and plots
├── scripts/                # utility scripts (e.g. mock data generation)
└── tests/                  # automated tests
```

## Installation

Tested on Python 3.10, TensorFlow 2.15, Windows and Linux.

**1. Create conda environment:**
```bash
conda create -n rfi-detection python=3.10
conda activate rfi-detection
```

**2. Install dependencies:**
```bash
conda install numpy=1.24.3
pip install tensorflow==2.15.0
conda install -c pytorch faiss-gpu
pip install -r requirements.txt
pip install -e ".[dev]"
```

**3. Clone external dependencies:**
```bash
git clone https://github.com/mesarcik/RFI-NLN
```

Then add RFI-NLN to your Python path. On Linux/Mac, add to `.bashrc`:
```bash
export PYTHONPATH="/path/to/RFI-NLN:$PYTHONPATH"
```

On Windows or Kaggle, add to your script or notebook:
```python
import sys
sys.path.insert(0, '/path/to/RFI-NLN')
```

**4. AOFlagger (optional, Linux only):**
```bash
apt-get install aoflagger
```

See https://gitlab.com/aroffringa/aoflagger for other platforms.

## Data

Raw data is not included in this repository due to file size constraints. See `data/README.md` for dataset descriptions.

To generate synthetic mock data for testing:
```bash
python scripts/create_mock_data.py
```

## Usage

Each method can be configured via its YAML file in `experiments/configs/` and run through the corresponding script in `experiments/`.

Example:
```bash
python experiments/run_cusum.py --config experiments/configs/cusum_luserna.yaml
```

On Kaggle, override the data path before running:
```python
import yaml

with open('experiments/configs/cusum_luserna.yaml') as f:
    cfg = yaml.safe_load(f)

cfg['data_path'] = '/kaggle/input/<your-dataset>/'

with open('experiments/configs/cusum_luserna.yaml', 'w') as f:
    yaml.dump(cfg, f)

!python experiments/run_cusum.py \
    --rfinln_path /kaggle/working/RFI-NLN \
    --data_path /kaggle/input/<your-dataset>/
```

## Methods

| Method | Type |
|---|---|
| CUSUM | Statistical |
| SumThreshold | Statistical |
| KNN | Machine Learning |
| Random Forest | Machine Learning |
| U-Net | Deep Learning |
| R-Net | Deep Learning |

## Results

Detailed results per method and dataset are available in `experiments/results/`. A summary of F1-scores and computational times is reported in the thesis document.

## Requirements

See `requirements.txt` for the full list of dependencies.

## Citation

If you use this code, please cite the associated thesis:
```
Nastasi, F. (2025). Statistical Methods for Radio Frequency Interference Detection
in Radioastronomical Spectrograms. MSc Thesis, University of Turin.
```

This work builds on:

- Mesarcik et al. (2022). RFI-NLN: Radio Frequency Interference detection using Neural-Latent Nearest Neighbours. https://github.com/mesarcik/RFI-NLN

- Offringa et al. AOFlagger: RFI Software. https://gitlab.com/aroffringa/aoflagger

## Acknowledgements

Observations were conducted at the Astronomical Observatory of Luserna San Giovanni. The LOFAR dataset was used as an external benchmark following the methodology described in the original paper.

## License

This project is released for academic purposes.