# Deep Learning — Satellite Spectra Classification

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arch-sci/dl-satellite-spectra/blob/main/dl-spectra.ipynb)

Multi-input / multi-output deep learning model for automatic analysis of satellite spectra, based on a CNES (French Space Agency) challenge. The model simultaneously predicts two binary labels — water presence and cloud cover — from high-dimensional spectral measurements and auxiliary physical features.

## Problem

Each observation consists of:
- **Spectral data**: 52 wavelength measurements across 3 channels (3000 × 52 × 3)
- **Auxiliary data**: 5 physical features (star/planet mass, radius, temperature, orbital distance)
- **Targets**: two binary labels (`water`, `cloud`) — both class-balanced at 50%

## Approach

### Feature Engineering
Raw features alone produced ~50% accuracy (no better than random). After researching exoplanet atmospheric science, I engineered domain-informed features:

| Feature | Formula | Physical meaning |
|---|---|---|
| `flux_received` | T⁴ / d² | Stellar radiation hitting the planet |
| `surface_gravity` | M_planet / R_star² | Atmosphere retention capacity |
| `star_density` | M_star / R_star³ | Stellar compactness |
| `planet_star_mass_ratio` | M_planet / M_star | Relative planetary mass |

Log transforms applied to 5 wide-range features → **14 total auxiliary features**

For spectral data: channel analysis revealed only Channel 1 carries discriminative signal (cross-sample variance). 6 spectral metrics extracted (std, IQR, kurtosis, peak count, depth, cross-channel correlation) → **20 total tabular features**

### Architecture — Keras Functional API
Two-branch multi-input model:
- **Spectral branch**: Conv1D (64 filters, k=3) → BatchNorm → Conv1D (32 filters) → GlobalAveragePooling1D
- **Tabular branch**: Dense(64) → BatchNorm → Dropout(0.3) → Dense(32)
- **Merged**: concatenated → Dense(64) → two separate sigmoid heads (water, cloud)

Two variants compared: Conv1D vs Flatten for the spectral branch.

Training: AdamW, binary cross-entropy, EarlyStopping (patience=15), max 150 epochs, batch size 64.

## Results

| Model | Water AUC | Cloud AUC | Water Acc | Cloud Acc |
|---|---|---|---|---|
| **Conv1D (selected)** | **0.78** | **0.77** | **~0.69** | **~0.69** |
| Flatten (baseline) | 0.58 | 0.58 | ~0.56 | ~0.56 |

Conv1D decisively outperforms the Flatten baseline — explicitly modelling local wavelength structure with convolutional filters is critical for extracting signal from the spectral sequence.
This project got a mark of 19/20.

## Stack
Python · numpy · pandas · scikit-learn · TensorFlow · Keras · matplotlib · seaborn · scipy
