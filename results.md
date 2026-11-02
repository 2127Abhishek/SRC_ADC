# SAR ADC results

## Each non-ideality alone

| Case | max|INL| | max|DNL| | SFDR (dB) | ENOB |
|---|---|---|---|---|
| Ideal | 0.00 | 0.00 | 83.4 | 10.00 |
| Cap mismatch only | 0.58 | 0.54 | 70.0 | 9.70 |
| Comparator offset only | 0.00 | 0.00 | 83.5 | 10.00 |
| kT/C noise only | 0.05 | 0.06 | 84.4 | 9.97 |
| Comparator noise only | 0.08 | 0.09 | 87.6 | 9.59 |
| All on | 0.64 | 0.52 | 69.9 | 9.39 |

## Single die, correction comparison

| Method | max|INL| (LSB) | max|DNL| (LSB) | SNDR (dB) | SFDR (dB) | ENOB (bits) |
|---|---|---|---|---|---|
| Uncorrected | 0.60 | 0.53 | 57.8 | 69.5 | 9.39 |
| LUT | 0.10 | 0.13 | 58.8 | 82.2 | 9.54 |
| ML-Linear | 0.15 | 0.16 | 58.8 | 80.5 | 9.54 |
| ML-GradBoost | 0.09 | 0.12 | 58.8 | 80.9 | 9.54 |

## Monte Carlo (20 dies)

| Method | max|INL| (LSB) | max|DNL| (LSB) | SFDR (dB) | ENOB (bits) |
|---|---|---|---|---|
| Uncorrected | 0.66 ± 0.15 | 0.59 ± 0.17 | 68.9 ± 4.1 | 9.33 ± 0.11 |
| LUT | 0.21 ± 0.09 | 0.31 ± 0.17 | 86.9 ± 1.1 | 9.55 ± 0.01 |
| ML-Linear | 0.34 ± 0.21 | 0.35 ± 0.20 | 85.2 ± 2.0 | 9.55 ± 0.01 |
| ML-GradBoost | 0.22 ± 0.11 | 0.31 ± 0.17 | 86.1 ± 1.5 | 9.55 ± 0.01 |
