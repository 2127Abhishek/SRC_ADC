# SAR ADC with ML-Based Nonlinearity Correction

A 10-bit SAR ADC behavioral model in Python (NumPy) with sample-and-hold, a binary-weighted
capacitive DAC, a comparator, and SAR logic. It models capacitor mismatch, comparator offset,
kT/C noise, and comparator noise, each with its own on/off switch.

**Characterization:** INL/DNL (ramp and sine histogram) and SNDR, SFDR, ENOB (FFT of a
coherently sampled sine).

**Correction:** a per-code LUT baseline vs ML regressors (linear regression, gradient
boosting) trained on bit-level features, evaluated on held-out signals over 10 Monte Carlo dies.

## Results (mean over 10 dies)

| Method | max INL (LSB) | SFDR (dB) | ENOB |
|---|---|---|---|
| Uncorrected | 0.65 | 68.9 | 9.33 |
| LUT | 0.20 | 86.7 | 9.55 |
| ML-GradBoost | 0.20 | 85.8 | 9.55 |

Correction removes deterministic mismatch error (INL about 3x lower, SFDR about +17 dB).
ENOB stays noise-limited near 9.55 bits because random comparator noise cannot be corrected.
ML matched the LUT but did not beat it.

## Run

pip install numpy matplotlib scikit-learn
python sar_adc.py
