"""
Regenerate PSTHs from the minimal dataset using the 2023 analog-MUA method.

Rules:
- Read-only inputs from /Volumes/MannySSD/lmc_project_v2_MINIMAL.
- No modification of existing repo code or dataset files.
- Outputs go to a new base directory; if it already exists, the script aborts.

Analog -> spike pipeline (per recording):
    noise = median(|signal|) / 0.6745
    spikes = signal < -3 * noise
    bin at 1 ms (10 kHz -> 10 samples/bin)
    smooth with Gaussian window (50 ms window, sigma = 5 ms)
    firing_rate_estimates in Hz (channels x time), here channels = 1
Stimulation:
    load onset/offset/stim_id from MUA/allData/timestamp_s.mat
PSTH:
    reuse NeuralAnalysis.calculate_psth_pre_post_and_plot_allgoodchannels
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import convolve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from neural_analysis_pkg.core import NeuralAnalysis  # type: ignore


DATA_ROOT = Path("/Volumes/MannySSD/lmc_project_v2_MINIMAL")
OUTPUT_BASE = Path("/Volumes/MannySSD/PSTH_regenerated_2025_run9")
SAMPLING_RATE = 10000  # Hz for analog_downsampled.dat
BIN_SIZE = 0.001  # seconds
GAUSS_WINDOW_MS = 50
GAUSS_SIGMA_MS = 5
N_CHANNELS = 32


def assert_safe_output_dir():
    if OUTPUT_BASE.exists():
        raise SystemExit(f"Output base {OUTPUT_BASE} already exists. Aborting to avoid overwrite.")


def gaussian_kernel(window_ms=GAUSS_WINDOW_MS, sigma_ms=GAUSS_SIGMA_MS, bin_size=BIN_SIZE) -> np.ndarray:
    window_samples = int(window_ms / (bin_size * 1000))  # 50 ms / 1 ms = 50 samples
    sigma_samples = sigma_ms / (bin_size * 1000)  # 5 ms / 1 ms = 5 samples
    half = window_samples // 2
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
    kernel /= kernel.sum()
    return kernel


def compute_firing_rate_from_analog(analog_path: Path) -> np.ndarray:
    """
    Load analog MUA, detect spikes via fixed threshold, bin to 1 ms, smooth with Gaussian.
    Returns (1, T) firing_rate_estimates in Hz.
    """
    signal = np.fromfile(analog_path, dtype=np.float32)
    noise = np.median(np.abs(signal)) / 0.6745
    spikes = signal < (-3 * noise)

    samples_per_bin = int(SAMPLING_RATE * BIN_SIZE)  # 10 samples
    n_full_bins = spikes.size // samples_per_bin
    trimmed = spikes[: n_full_bins * samples_per_bin]
    binned_counts = trimmed.reshape(n_full_bins, samples_per_bin).sum(axis=1)
    firing_rate_hz = binned_counts / BIN_SIZE  # counts per 1 ms -> Hz

    kernel = gaussian_kernel()
    smoothed = convolve(firing_rate_hz, kernel, mode="same")
    # Repeat across channels to satisfy downstream PSTH plotting expectations (32 ch).
    return np.tile(smoothed, (N_CHANNELS, 1))


def load_stim_df(recording_name: str, timestamp_path: Path) -> pd.DataFrame:
    mat = loadmat(timestamp_path)
    ts = mat["timestamp_s"]
    # Columns: onset, offset, stim_id; drop any invalid rows (e.g., huge negative placeholders)
    ts = ts[np.isfinite(ts).all(axis=1)]
    return pd.DataFrame(
        {
            "recording_name": recording_name,
            "onset_times": ts[:, 0],
            "offset_times": ts[:, 1],
            "stimulation_ids": ts[:, 2].astype(int),
        }
    )


def remap_recordings():
    """
    Build in-memory recording_results_df and stim df using the minimal dataset.
    One analog file per recording; channels=1 with good_channels=[0].
    """
    rec_rows = []
    stim_rows = []
    for group in ["LED", "Whisker"]:
        rr_path = DATA_ROOT / group / "SpikeStuff" / "recording_results.csv"
        df = pd.read_csv(rr_path)
        for _, row in df.iterrows():
            recording_name = row["recording_name"]
            # Locate the recording folder under the minimal dataset
            rec_dir = DATA_ROOT / group / "SpikeStuff" / row["group_name"] / recording_name
            analog_candidates = list((rec_dir / "AnalogSignal").glob("*analog_downsampled.dat"))
            if not analog_candidates:
                print(f"Skipping {recording_name}: no analog_downsampled.dat found in {rec_dir}")
                continue
            analog_path = analog_candidates[0]
            timestamp_path = rec_dir / "MUA" / "allData" / "timestamp_s.mat"
            if not timestamp_path.exists():
                print(f"Skipping {recording_name}: missing {timestamp_path}")
                continue

            rec_rows.append(
                {
                    "recording_name": recording_name,
                    "group_name": row["group_name"],
                    "good_channels": list(range(N_CHANNELS)),  # fill all channels to satisfy plotting grid
                    "noisy_channels": [],
                    "has_required_assets": True,
                    "analog_path": str(analog_path),
                    "mua_data_path": str(analog_path),
                    "downsampled_path": str(analog_path),
                    "timestamp_path": str(timestamp_path),
                    "spike_times_path": None,
                    "spike_times_path": None,
                }
            )
            stim_rows.append(load_stim_df(recording_name, timestamp_path))

    recording_results_df = pd.DataFrame(rec_rows)
    stimulation_data_df = pd.concat(stim_rows, ignore_index=True) if stim_rows else pd.DataFrame()
    return recording_results_df, stimulation_data_df


def main():
    assert_safe_output_dir()
    recording_results_df, stim_df = remap_recordings()
    if recording_results_df.empty:
        raise SystemExit("No recordings with required assets found in minimal dataset.")

    analysis = NeuralAnalysis(file_path=DATA_ROOT, n_channels=N_CHANNELS, sampling_rate=SAMPLING_RATE)
    analysis.recording_results_df = recording_results_df
    analysis.stimulation_data_df = stim_df

    for _, row in recording_results_df.iterrows():
        rec_name = row["recording_name"]
        analog_path = Path(row["analog_path"])
        firing_rates = compute_firing_rate_from_analog(analog_path)
        save_path = OUTPUT_BASE / "whisker_psths_prevspost_regenerated" / f"{rec_name}_psth_prevspost_REGEN.svg"
        if save_path.exists():
            print(f"Skipping {rec_name}: output already exists at {save_path}")
            continue
        print(f"Processing {rec_name}: analog {analog_path}")
        analysis.calculate_psth_pre_post_and_plot_allgoodchannels(
            rec_name,
            firing_rates,
            base_dir=OUTPUT_BASE,
            bin_size=BIN_SIZE,
            pre_trials=30,
            post_trials=30,
        )


if __name__ == "__main__":
    main()
