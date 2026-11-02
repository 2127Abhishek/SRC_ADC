#!/usr/bin/env python3
"""
10-bit SAR ADC behavioral model with non-idealities, INL/DNL + SNR/SFDR/ENOB
characterization, and nonlinearity correction (LUT baseline vs ML regressors).

Usage:
    python sar_adc.py                  # full run (single die + Monte Carlo)
    python sar_adc.py --dies 5         # faster Monte Carlo
    python sar_adc.py --sigma-unit 0.01

Outputs (in ./sar_adc_results): PNG figures + results.md
Requires: numpy, matplotlib, scikit-learn
"""
import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

K_B = 1.380649e-23

# Dynamic test setup (coherent sampling): fin = M * fs / NFFT, M odd -> coprime with NFFT
NFFT = 16384
M_CYCLES = 1013
AMP_FRAC = 0.95  # sine amplitude as a fraction of full-scale half-range


# --------------------------------------------------------------------------
# 1) ADC model
# --------------------------------------------------------------------------
class SARADC:
    """Charge-redistribution SAR ADC, behavioral.

    Cap array: C_k = 2^k * Cu (k = 0..N-1) + one dummy Cu, so ideal weights are 2^k / 2^N.
    Mismatch: sigma(C_k)/C_k = sigma_unit / sqrt(2^k)  (Pelgrom-style area scaling).
    Each non-ideality has its own on/off switch.
    """

    def __init__(self, n_bits=10, vref=1.0, c_unit=1e-15, sigma_unit=0.02,
                 offset_sigma=1e-3, comp_noise_lsb=0.25, temp=300.0,
                 mismatch=True, offset=True, ktc=True, comp_noise=True, seed=0):
        self.N = n_bits
        self.vref = vref
        self.lsb = vref / 2 ** n_bits
        rng = np.random.default_rng(seed)
        k = np.arange(n_bits)

        # draw all randoms first so toggling a switch doesn't change the other draws
        z_caps = rng.standard_normal(n_bits)
        z_dummy = rng.standard_normal()
        z_off = rng.standard_normal()

        c_ideal = c_unit * 2.0 ** k
        if mismatch:
            c = c_ideal * (1 + sigma_unit / np.sqrt(2.0 ** k) * z_caps)
            c_dummy = c_unit * (1 + sigma_unit * z_dummy)
        else:
            c = c_ideal
            c_dummy = c_unit

        self.c_total = c.sum() + c_dummy
        self.weights = c / self.c_total  # fraction of Vref switched by each bit
        self.offset = offset_sigma * z_off if offset else 0.0  # input-referred comparator offset (V)
        self.ktc_sigma = np.sqrt(K_B * temp / self.c_total) if ktc else 0.0  # sampled kT/C noise (V rms)
        self.comp_sigma = comp_noise_lsb * self.lsb if comp_noise else 0.0  # per-decision noise (V rms)

    def convert(self, vin, rng):
        """Vectorized conversion of an array of input voltages -> integer codes."""
        v = np.asarray(vin, dtype=float)
        if self.ktc_sigma:
            v = v + rng.normal(0.0, self.ktc_sigma, v.shape)  # sample-and-hold noise
        v = v + self.offset
        code = np.zeros(v.shape, dtype=np.int64)
        dac = np.zeros(v.shape)
        for k in range(self.N - 1, -1, -1):  # MSB -> LSB
            trial = dac + self.weights[k] * self.vref
            x = v
            if self.comp_sigma:
                x = v + rng.normal(0.0, self.comp_sigma, v.shape)  # comparator noise each decision
            keep = x >= trial
            dac = np.where(keep, trial, dac)
            code |= keep.astype(np.int64) << k
        return code


def bit_matrix(codes, n_bits):
    return ((codes[:, None] >> np.arange(n_bits)) & 1).astype(float)


# --------------------------------------------------------------------------
# 2) Static characterization (INL / DNL)
# --------------------------------------------------------------------------
def ramp_static(codes, n_bits):
    """Histogram method on a slow ramp. Returns DNL, INL (LSB) for codes 1..2^N-2."""
    n = 2 ** n_bits
    counts = np.bincount(codes, minlength=n).astype(float)
    inner = counts[1:-1]  # end codes are clipped by the ramp overdrive
    dnl = inner / inner.mean() - 1.0
    inl = np.cumsum(dnl)
    inl = inl - np.linspace(inl[0], inl[-1], len(inl))  # endpoint-fit: removes offset and gain error
    return dnl, inl


def sine_hist_static(codes, n_bits):
    """Sine histogram (Doernberg) method: transition levels from the cumulative histogram,
    so amplitude and offset of the stimulus need not be known."""
    n = 2 ** n_bits
    counts = np.bincount(codes, minlength=n).astype(float)
    cum = np.cumsum(counts)[:-1] / counts.sum()  # n-1 transitions
    t = -np.cos(np.pi * cum)
    lsb = (t[-1] - t[0]) / (len(t) - 1)
    dnl = np.diff(t) / lsb - 1.0
    line = np.linspace(t[0], t[-1], len(t))
    inl = (t - line) / lsb
    return dnl, inl


# --------------------------------------------------------------------------
# 3) Dynamic characterization (SNDR / SNR / SFDR / ENOB)
# --------------------------------------------------------------------------
def dynamic_metrics(codes, n_bits, m=M_CYCLES, nfft=NFFT, amp_frac=AMP_FRAC, n_harm=9):
    x = codes.astype(float)
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x)) ** 2  # coherent sampling -> no window needed
    p[0] = 0.0
    p_sig = p[m]

    harm_bins = set()
    for h in range(2, n_harm + 1):
        b = (h * m) % nfft
        if b > nfft // 2:
            b = nfft - b
        if b != m:
            harm_bins.add(b)

    p_other = p.copy()
    p_other[m] = 0.0
    p_nad = p_other.sum()  # noise + distortion
    p_h = sum(p_other[b] for b in harm_bins)
    p_noise = p_nad - p_h

    sndr = 10 * np.log10(p_sig / p_nad)
    snr = 10 * np.log10(p_sig / p_noise)
    sfdr = 10 * np.log10(p_sig / p_other.max())
    enob = (sndr - 1.76 + 20 * np.log10(1.0 / amp_frac)) / 6.02  # corrected for sub-full-scale input
    spec_db = 10 * np.log10(np.maximum(p, 1e-30) / p_sig)
    return dict(sndr=sndr, snr=snr, sfdr=sfdr, enob=enob, spec=spec_db)


# --------------------------------------------------------------------------
# 4) Correction: LUT baseline and ML regressors
# --------------------------------------------------------------------------
class LUTCorrector:
    """Per-code average of (true input - raw code center), learned from training data."""

    name = "LUT"

    def __init__(self, n_bits):
        self.n = 2 ** n_bits
        self.n_bits = n_bits

    def fit(self, codes, true_cont):
        err = true_cont - (codes + 0.5)
        sums = np.bincount(codes, weights=err, minlength=self.n)
        cnts = np.bincount(codes, minlength=self.n)
        self.table = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)
        return self

    def apply(self, codes):
        """Real-valued corrected output in LSB (a wider digital word, as real calibration gives)."""
        return np.clip(codes + 0.5 + self.table[codes], 0, self.n)


class MLCorrector:
    """Regressor on the individual bit values -> predicted error in LSB."""

    def __init__(self, n_bits, model, name):
        self.n_bits = n_bits
        self.n = 2 ** n_bits
        self.model = model
        self.name = name

    def fit(self, codes, true_cont):
        err = true_cont - (codes + 0.5)
        self.model.fit(bit_matrix(codes, self.n_bits), err)
        return self

    def apply(self, codes):
        err = self.model.predict(bit_matrix(codes, self.n_bits))
        return np.clip(codes + 0.5 + err, 0, self.n)


def make_correctors(adc, rng, n_train, gb_iter):
    """Calibrate on THIS die with random uniform inputs. Test signals are different (ramp, sine)."""
    vin = rng.uniform(0.0, adc.vref, n_train)
    codes = adc.convert(vin, rng)
    true_cont = vin / adc.vref * 2 ** adc.N
    return [
        LUTCorrector(adc.N).fit(codes, true_cont),
        MLCorrector(adc.N, LinearRegression(), "ML-Linear").fit(codes, true_cont),
        MLCorrector(adc.N, HistGradientBoostingRegressor(max_iter=gb_iter, learning_rate=0.1, random_state=0),
                    "ML-GradBoost").fit(codes, true_cont),
    ]


# --------------------------------------------------------------------------
# 5) Evaluation
# --------------------------------------------------------------------------
def evaluate(adc, correctors, rng, spc=256, trim=8):
    """Convert once (same noise realization), then apply every corrector to the same raw codes.

    INL/DNL come from the transfer curve: mean(output - true input) per 1-LSB input bin, endpoint-fit
    removed (offset and gain error excluded). DNL = first difference of INL. This works for real-valued
    corrected outputs, where a code histogram is undefined.
    Dynamic test: corrected output is a 12-bit word (floor(est*4)/4); uncorrected is the raw 10-bit code.
    """
    n = 2 ** adc.N
    ramp = (np.arange(n * spc) + 0.5) / (n * spc) * adc.vref
    true_cont = ramp / adc.vref * n
    t = np.arange(NFFT)
    sine = adc.vref / 2 * (1 + AMP_FRAC * np.sin(2 * np.pi * M_CYCLES * t / NFFT))
    ramp_codes = adc.convert(ramp, rng)
    sine_codes = adc.convert(sine, rng)

    out = {}
    for name, c in [("Uncorrected", None)] + [(c.name, c) for c in correctors]:
        if c is None:
            ramp_est = ramp_codes + 0.5
            sine_out = sine_codes.astype(float)
        else:
            ramp_est = c.apply(ramp_codes)
            sine_out = np.floor(c.apply(sine_codes) * 4) / 4
        err = (ramp_est - true_cont).reshape(n, spc).mean(axis=1)[trim:-trim]
        inl = err - np.linspace(err[0], err[-1], len(err))
        dnl = np.diff(inl)
        dyn = dynamic_metrics(sine_out, adc.N)
        out[name] = dict(inl=inl, dnl=dnl, inl_max=np.abs(inl).max(), dnl_max=np.abs(dnl).max(), **dyn)
    return out


def fmt_table(results):
    lines = ["| Method | max|INL| (LSB) | max|DNL| (LSB) | SNDR (dB) | SFDR (dB) | ENOB (bits) |",
             "|---|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['inl_max']:.2f} | {r['dnl_max']:.2f} | "
                     f"{r['sndr']:.1f} | {r['sfdr']:.1f} | {r['enob']:.2f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 6) Main flow
# --------------------------------------------------------------------------
def sanity_checks(rng):
    print("== Sanity: all non-idealities OFF ==")
    ideal = SARADC(mismatch=False, offset=False, ktc=False, comp_noise=False)
    vin = np.linspace(0, ideal.vref, 2 ** 20, endpoint=False)
    codes = ideal.convert(vin, rng)
    expected = np.floor(vin / ideal.vref * 2 ** ideal.N).astype(np.int64)
    print(f"  ramp: codes == ideal floor(vin/LSB) everywhere: {bool(np.all(codes == expected))}")
    r = evaluate(ideal, [], rng, spc=256)["Uncorrected"]
    print(f"  ideal ENOB = {r['enob']:.2f} bits, SNDR = {r['sndr']:.1f} dB (expect ~10 bits / ~62 dB)")
    print(f"  ideal max|DNL| = {r['dnl_max']:.3f} LSB (estimation noise only)\n")


def ablation(args, rng):
    print("== Effect of each non-ideality alone ==")
    rows = []
    cases = [("Ideal", dict(mismatch=False, offset=False, ktc=False, comp_noise=False)),
             ("Cap mismatch only", dict(mismatch=True, offset=False, ktc=False, comp_noise=False)),
             ("Comparator offset only", dict(mismatch=False, offset=True, ktc=False, comp_noise=False)),
             ("kT/C noise only", dict(mismatch=False, offset=False, ktc=True, comp_noise=False)),
             ("Comparator noise only", dict(mismatch=False, offset=False, ktc=False, comp_noise=True)),
             ("All on", dict())]
    for label, flags in cases:
        adc = SARADC(sigma_unit=args.sigma_unit, seed=args.seed, **flags)
        r = evaluate(adc, [], rng)["Uncorrected"]
        rows.append(f"| {label} | {r['inl_max']:.2f} | {r['dnl_max']:.2f} | {r['sfdr']:.1f} | {r['enob']:.2f} |")
    table = ("| Case | max|INL| | max|DNL| | SFDR (dB) | ENOB |\n|---|---|---|---|---|\n" + "\n".join(rows))
    print(table + "\n")
    return table


def plot_single_die(res, adc, rng, outdir):
    n = 2 ** adc.N
    codes = np.arange(8, n - 8)  # transfer-curve INL (trimmed ends)
    dcodes = codes[1:]

    # Static cross-check on the uncorrected die: ramp histogram and sine histogram
    ramp = np.linspace(-0.005 * adc.vref, 1.005 * adc.vref, n * 512)
    dnl_r, inl_r = ramp_static(adc.convert(ramp, rng), adc.N)
    u = np.linspace(0, 1, 2 ** 21, endpoint=False)
    sine = adc.vref / 2 * (1 + 1.02 * np.sin(2 * np.pi * u))  # slight overdrive so end codes get hit
    dnl_s, inl_s = sine_hist_static(adc.convert(sine, rng), adc.N)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(codes, res["Uncorrected"]["inl"], lw=0.8, label="transfer curve")
    ax[0].plot(np.arange(1, n - 1), inl_r, lw=0.8, alpha=0.7, label="ramp histogram")
    ax[0].plot(np.arange(len(inl_s)), inl_s, lw=0.8, alpha=0.7, label="sine histogram")
    ax[0].set_ylabel("INL (LSB)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(dcodes, res["Uncorrected"]["dnl"], lw=0.6, label="transfer curve")
    ax[1].plot(np.arange(1, n - 1), dnl_r, lw=0.6, alpha=0.7, label="ramp histogram")
    ax[1].plot(np.arange(len(dnl_s)), dnl_s, lw=0.6, alpha=0.7, label="sine histogram")
    ax[1].set_ylabel("DNL (LSB)"); ax[1].set_xlabel("Code"); ax[1].grid(alpha=0.3)
    fig.suptitle("Uncorrected static linearity")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "static_uncorrected.png"), dpi=130); plt.close(fig)
    print(f"  uncorrected cross-check, max|INL|/max|DNL| (LSB): transfer {res['Uncorrected']['inl_max']:.2f}/"
          f"{res['Uncorrected']['dnl_max']:.2f}, ramp-hist {np.abs(inl_r).max():.2f}/{np.abs(dnl_r).max():.2f}, "
          f"sine-hist {np.abs(inl_s).max():.2f}/{np.abs(dnl_s).max():.2f}")

    # INL before / after
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, r in res.items():
        ax.plot(codes, r["inl"], lw=0.8, label=f"{name} (max {r['inl_max']:.2f})")
    ax.set_xlabel("Code"); ax.set_ylabel("INL (LSB)"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("INL before / after correction (held-out ramp)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "inl_before_after.png"), dpi=130); plt.close(fig)

    # Spectra before / after
    fig, ax = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey=True)
    freqs = np.arange(NFFT // 2 + 1) / NFFT
    for a, (name, r) in zip(ax.ravel(), res.items()):
        a.plot(freqs, r["spec"], lw=0.4)
        a.set_title(f"{name}: SFDR {r['sfdr']:.1f} dB, ENOB {r['enob']:.2f}")
        a.set_ylim(-110, 5); a.grid(alpha=0.3)
    for a in ax[1]: a.set_xlabel("f / fs")
    for a in ax[:, 0]: a.set_ylabel("dBc")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "spectrum_before_after.png"), dpi=130); plt.close(fig)


def monte_carlo(args, outdir):
    print(f"== Monte Carlo over {args.dies} dies (each die calibrated on its own training data) ==")
    methods = ["Uncorrected", "LUT", "ML-Linear", "ML-GradBoost"]
    stats = {m: {"inl_max": [], "dnl_max": [], "sfdr": [], "enob": []} for m in methods}
    for d in range(args.dies):
        t0 = time.time()
        adc = SARADC(sigma_unit=args.sigma_unit, seed=1000 + d)
        rng = np.random.default_rng(5000 + d)
        corr = make_correctors(adc, rng, n_train=2 ** 17, gb_iter=100)
        res = evaluate(adc, corr, rng, spc=256)
        for m in methods:
            for key in stats[m]:
                stats[m][key].append(res[m][key])
        print(f"  die {d + 1}/{args.dies} done ({time.time() - t0:.1f}s)")

    lines = ["| Method | max|INL| (LSB) | max|DNL| (LSB) | SFDR (dB) | ENOB (bits) |", "|---|---|---|---|---|"]
    for m in methods:
        s = stats[m]
        lines.append(f"| {m} | {np.mean(s['inl_max']):.2f} ± {np.std(s['inl_max']):.2f} | "
                     f"{np.mean(s['dnl_max']):.2f} ± {np.std(s['dnl_max']):.2f} | "
                     f"{np.mean(s['sfdr']):.1f} ± {np.std(s['sfdr']):.1f} | "
                     f"{np.mean(s['enob']):.2f} ± {np.std(s['enob']):.2f} |")
    table = "\n".join(lines)
    print("\n" + table + "\n")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, key, label in zip(ax, ["inl_max", "sfdr", "enob"], ["max|INL| (LSB)", "SFDR (dB)", "ENOB (bits)"]):
        a.boxplot([stats[m][key] for m in methods])
        a.set_xticklabels(methods)
        a.set_ylabel(label); a.grid(alpha=0.3); a.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Monte Carlo, {args.dies} dies, unit-cap sigma = {args.sigma_unit * 100:.1f}%")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "monte_carlo.png"), dpi=130); plt.close(fig)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dies", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-unit", type=float, default=0.02, help="unit-cap relative mismatch sigma")
    ap.add_argument("--out", default="sar_adc_results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sanity_checks(rng)
    abl = ablation(args, rng)

    print("== Single die: correction comparison (held-out ramp + sine) ==")
    adc = SARADC(sigma_unit=args.sigma_unit, seed=args.seed)
    print(f"  kT/C noise = {adc.ktc_sigma * 1e6:.1f} uV rms ({adc.ktc_sigma / adc.lsb:.3f} LSB), "
          f"comparator noise = {adc.comp_sigma * 1e6:.0f} uV rms, offset = {adc.offset * 1e3:.2f} mV")
    corr = make_correctors(adc, rng, n_train=2 ** 18, gb_iter=200)
    res = evaluate(adc, corr, rng, spc=512)
    single = fmt_table(res)
    print(single + "\n")
    plot_single_die(res, adc, rng, args.out)

    mc = monte_carlo(args, args.out)

    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write("# SAR ADC results\n\n## Each non-ideality alone\n\n" + abl +
                "\n\n## Single die, correction comparison\n\n" + single +
                f"\n\n## Monte Carlo ({args.dies} dies)\n\n" + mc + "\n")
    print(f"Saved figures and results.md to ./{args.out}/")


if __name__ == "__main__":
    main()