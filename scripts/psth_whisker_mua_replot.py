"""
Re-run PSTH plotting using MUA-only (no spike_times) for recordings in the SSD copy.
- Reads from /Volumes/MannySSD/lmc_project_v2_MINIMAL
- Fixes path prefixes to the SSD root (Whisker/ and LED/ preserved)
- Computes firing rates from temp_wh_MUA.npy (10 kHz) via the historical method:
  noise = median(|signal|)/0.6745; spikes = signal < -3*noise; bin 1 ms; Gaussian smooth 50 ms window, sigma=5 ms.
- Calls calculate_psth_pre_post_and_plot_allgoodchannels with a fresh base_dir to avoid overwrite.
- Saves processed_psths_dict.pkl and recording_results_df.pkl in the output folder.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import convolve

import matplotlib
matplotlib.use('Agg')

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
from neural_analysis_pkg.core import NeuralAnalysis  # type: ignore

DATA_ROOT = Path('/Volumes/MannySSD/lmc_project_v2_MINIMAL')
OLD_ROOTS = [
    '/home/cresp1el-local/Documents/MATLAB/Data/lmc_project_v2/Whisker/',
    '/home/cresp1el-local/Documents/MATLAB/Data/lmc_project_v2/LED/',
    '/home/cresp1el-local/Documents/MATLAB/Data/lmc_project_v2/',
]
NEW_ROOT = str(DATA_ROOT) + '/'
OUTPUT_BASE = Path('/Volumes/MannySSD/PSTH_replot_from_saved_dicts/run13')
SAMPLING_RATE = 10000
BIN_SIZE = 0.001
GAUSS_WINDOW_MS = 50
GAUSS_SIGMA_MS = 5


def gaussian_kernel(window_ms=GAUSS_WINDOW_MS, sigma_ms=GAUSS_SIGMA_MS, bin_size=BIN_SIZE):
    window_samples = int(window_ms / (bin_size * 1000))
    sigma_samples = sigma_ms / (bin_size * 1000)
    half = window_samples // 2
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (x / sigma_samples) ** 2)
    k /= k.sum()
    return k


def firing_rate_from_mua(mua_path: Path) -> np.ndarray:
    sig = np.load(mua_path)
    if sig.ndim == 1:
        sig = sig[:, None]
    sig = np.nan_to_num(sig, nan=0.0)
    n_samples, n_ch = sig.shape
    noise = np.median(np.abs(sig), axis=0) / 0.6745
    thresh = -3 * noise
    spikes = sig < thresh
    samples_per_bin = int(SAMPLING_RATE * BIN_SIZE)
    n_full_bins = n_samples // samples_per_bin
    spikes = spikes[: n_full_bins * samples_per_bin, :]
    spikes = spikes.reshape(n_full_bins, samples_per_bin, n_ch).sum(axis=1)
    fr_hz = spikes / BIN_SIZE  # counts per ms -> Hz
    kern = gaussian_kernel()
    smoothed = np.vstack([convolve(fr_hz[:, ch], kern, mode='same') for ch in range(n_ch)])
    return smoothed  # shape (channels, time)


def fix_paths(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['mua_data_path', 'downsampled_path']:
        if col in df.columns:
            s = df[col].astype(str)
            for old in OLD_ROOTS:
                s = s.str.replace(old, NEW_ROOT, regex=False)
            df[col] = s
    return df


def main():
    if OUTPUT_BASE.exists():
        raise SystemExit(f"Output folder {OUTPUT_BASE} already exists; stopping to avoid overwrite.")

    analysis = NeuralAnalysis(file_path=DATA_ROOT)
    analysis.extract_stimulation_data()
    analysis.recording_results_df = fix_paths(analysis.recording_results_df)

    # Mark has_required_assets based on MUA existence
    def has_assets(row):
        p = Path(row.get('mua_data_path', ''))
        return p.exists()
    analysis.recording_results_df['has_required_assets'] = analysis.recording_results_df.apply(has_assets, axis=1)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=False)

    processed_psths = {}
    for rec in analysis.recording_results_df['recording_name']:
        row = analysis.recording_results_df.loc[analysis.recording_results_df['recording_name'] == rec].iloc[0]
        mua_path = Path(row['mua_data_path'])
        if not mua_path.exists():
            print(f"Skipping {rec}: missing MUA {mua_path}")
            continue
        print(f"Processing {rec} from {mua_path}")
        fr = firing_rate_from_mua(mua_path)
        processed = analysis.calculate_psth_pre_post_and_plot_allgoodchannels(
            rec,
            fr,
            base_dir=OUTPUT_BASE,
            pre_trials=30,
            post_trials=30,
        )
        processed_psths[rec] = processed[rec]

    # Save for reuse
    import pickle
    with open(OUTPUT_BASE / 'processed_psths_dict.pkl', 'wb') as f:
        pickle.dump(processed_psths, f)
    analysis.recording_results_df.to_pickle(OUTPUT_BASE / 'recording_results_df.pkl')

    print(f"Done. Saved to {OUTPUT_BASE}")


if __name__ == '__main__':
    main()
