
"""
Build per-recording caches (firing rates, spikes, metadata) from the minimal analog dataset.

Inputs (read-only):
- /Volumes/MannySSD/lmc_project_v2_MINIMAL/<LED|Whisker>/SpikeStuff/<group>/<recording>/
    - AnalogSignal/*analog_downsampled.dat   (10 kHz)
    - MUA/allData/timestamp_s.mat            (onset, offset, stim_id)
    - recording_results.csv                  (for group/recording names)

Method (per recording, 2023 analog pipeline):
- noise = median(|signal|) / 0.6745
- spikes = signal < -3 * noise
- bin at 1 ms (10 samples)
- smooth with Gaussian window 50 ms, sigma 5 ms
- firing_rate_hz = smoothed (Hz)
- spike times (relative to onset) collected per trial

Outputs (per recording, under a fresh base directory):
- meta.parquet
- trials.parquet
- firing_tensor.npz  (firing_hz [channels x trials x time_bins], time_ms)
- spikes_relative.npz (object array spikes_rel[ch, trial] = spike times (s) relative to onset)
- psth_mean_sem.npz (mean/sem by stim_id across trials, per channel)

Safety:
- If the chosen output base exists, abort. No writes into the dataset.
- Uses n_channels_out=1 (analog trace treated as single channel).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import convolve

DATA_ROOT = Path("/Volumes/MannySSD/lmc_project_v2_MINIMAL")
# Default output base set to latest successful run; change for new runs.
DEFAULT_OUTPUT_BASE = Path("/Volumes/MannySSD/PSTH_regenerated_cache_lmc_run3")
SAMPLING_RATE = 10_000  # Hz
BIN_SIZE = 0.001        # seconds
PRE_MS = 500
POST_MS = 1000
GAUSS_WINDOW_MS = 50
GAUSS_SIGMA_MS = 5
N_CHANNELS_OUT = 1  # analog is single-channel; extend if needed


class SpikeHistoryCacheBuilder:
    def __init__(self, data_root=DATA_ROOT, output_base=DEFAULT_OUTPUT_BASE):
        self.data_root = Path(data_root)
        self.output_base = Path(output_base)
        self.kernel = self._gaussian_kernel()
        self.time_ms = np.arange(-PRE_MS, POST_MS, BIN_SIZE * 1000)

    def summarize_inputs(self):
        """
        Return a dataframe of candidate recordings with key paths and a quick availability flag.
        Useful for debugging before running the cache build.
        """
        rec_df, _ = self._gather_recordings()
        if rec_df.empty:
            print("No recordings found under", self.data_root)
            return rec_df
        rec_df = rec_df.assign(
            analog_exists=rec_df["analog_path"].apply(lambda p: Path(p).exists()),
            timestamp_exists=rec_df["timestamp_path"].apply(lambda p: Path(p).exists()),
        )
        print(rec_df[["recording_name", "group_name", "analog_exists", "timestamp_exists"]])
        return rec_df

    def summarize_output(self, base: Path | None = None, max_records: int = 5):
        """
        Inspect an existing cache directory and print shapes/contents for sanity checking.
        Does not modify anything.
        """
        base = Path(base) if base else self.output_base
        if not base.exists():
            print(f"Output base {base} does not exist.")
            return
        rec_dirs = sorted(base.glob("*/*"))
        print(f"Found {len(rec_dirs)} recording caches under {base}")
        for rec_dir in rec_dirs[:max_records]:
            self.inspect_recording_cache(rec_dir, verbose=True)
        if len(rec_dirs) > max_records:
            print(f"...skipping {len(rec_dirs) - max_records} more")

    def inspect_recording_cache(self, rec_dir: Path, verbose: bool = False):
        """
        Load one recording cache and return a summary dict of types/shapes/keys.
        Set verbose=True to print a human-readable table for debugging.
        """
        rec_dir = Path(rec_dir)
        meta_path = rec_dir / "meta.parquet"
        trials_path = rec_dir / "trials.parquet"
        firing_path = rec_dir / "firing_tensor.npz"
        spikes_path = rec_dir / "spikes_relative.npz"
        psth_path = rec_dir / "psth_mean_sem.npz"

        summary = {"rec_dir": str(rec_dir)}

        meta = pd.read_parquet(meta_path) if meta_path.exists() else None
        trials = pd.read_parquet(trials_path) if trials_path.exists() else None
        firing = np.load(firing_path) if firing_path.exists() else None
        # Backward compatibility for object arrays saved with newer numpy (numpy._core)
        spikes = None
        if spikes_path.exists():
            import sys as _sys
            _sys.modules.setdefault('numpy._core', np.core)
            spikes = np.load(spikes_path, allow_pickle=True)
        psth = np.load(psth_path) if psth_path.exists() else None

        if meta is not None:
            summary["meta"] = {
                "shape": meta.shape,
                "dtypes": meta.dtypes.to_dict(),
                "values": meta.to_dict(orient="records"),
            }
        if trials is not None:
            summary["trials"] = {
                "shape": trials.shape,
                "columns": list(trials.columns),
                "dtypes": trials.dtypes.to_dict(),
                "stim_ids": trials["stim_id"].unique().tolist(),
            }
        if firing is not None:
            fhz = firing["firing_hz"]
            tms = firing["time_ms"]
            summary["firing_tensor"] = {
                "shape": fhz.shape,
                "dtype": str(fhz.dtype),
                "time_ms_len": tms.shape,
                "time_ms_range": (float(tms.min()), float(tms.max())),
                "stats": {
                    "min": float(fhz.min()),
                    "max": float(fhz.max()),
                    "mean": float(fhz.mean()),
                },
            }
        if spikes is not None:
            srel = spikes["spikes_rel"]
            example_len = 0
            if srel.size:
                try:
                    example_len = len(srel.flatten()[0])
                except Exception:
                    example_len = None
            summary["spikes_rel"] = {
                "shape": srel.shape,
                "dtype": str(srel.dtype),
                "example_length": example_len,
            }
        if psth is not None:
            summary["psth_arrays"] = {k: psth[k].shape for k in psth.files}

        summary["files_present"] = {
            "meta.parquet": meta_path.exists(),
            "trials.parquet": trials_path.exists(),
            "firing_tensor.npz": firing_path.exists(),
            "spikes_relative.npz": spikes_path.exists(),
            "psth_mean_sem.npz": psth_path.exists(),
        }

        if verbose:
            print(f"\n{rec_dir.name}:")
            for key, val in summary.items():
                if key == "rec_dir":
                    continue
                print(f"  {key}: {val}")
        return summary

    def _gaussian_kernel(self):
        window_samples = int(GAUSS_WINDOW_MS / (BIN_SIZE * 1000))
        sigma_samples = GAUSS_SIGMA_MS / (BIN_SIZE * 1000)
        half = window_samples // 2
        x = np.arange(-half, half + 1)
        k = np.exp(-0.5 * (x / sigma_samples) ** 2)
        return k / k.sum()

    def _compute_firing_and_spikes(self, analog_path: Path):
        signal = np.fromfile(analog_path, dtype=np.float32)
        noise = np.median(np.abs(signal)) / 0.6745
        spikes_mask = signal < (-3 * noise)

        samples_per_bin = int(SAMPLING_RATE * BIN_SIZE)  # 10 samples
        n_full_bins = spikes_mask.size // samples_per_bin
        if n_full_bins == 0:
            raise ValueError(f"No full bins for {analog_path}")
        trimmed = spikes_mask[: n_full_bins * samples_per_bin]
        binned_counts = trimmed.reshape(n_full_bins, samples_per_bin).sum(axis=1)
        firing_hz = binned_counts / BIN_SIZE  # counts per 1 ms -> Hz
        smoothed = convolve(firing_hz, self.kernel, mode="same")

        spike_times_abs = np.nonzero(spikes_mask)[0] / SAMPLING_RATE
        return smoothed, spike_times_abs

    def _load_stim_df(self, recording_name: str, timestamp_path: Path) -> pd.DataFrame:
        mat = loadmat(timestamp_path)
        ts = mat["timestamp_s"]
        ts = ts[np.isfinite(ts).all(axis=1)]
        return pd.DataFrame(
            {
                "recording_name": recording_name,
                "onset_times": ts[:, 0],
                "offset_times": ts[:, 1],
                "stimulation_ids": ts[:, 2].astype(int),
            }
        )

    def _gather_recordings(self):
        rec_rows = []
        stim_rows = []
        for group in ["LED", "Whisker"]:
            rr_path = self.data_root / group / "SpikeStuff" / "recording_results.csv"
            if not rr_path.exists():
                continue
            df = pd.read_csv(rr_path)
            for _, row in df.iterrows():
                recording_name = row["recording_name"]
                rec_dir = self.data_root / group / "SpikeStuff" / row["group_name"] / recording_name
                analog_candidates = list((rec_dir / "AnalogSignal").glob("*analog_downsampled.dat"))
                if not analog_candidates:
                    print(f"Skip {recording_name}: no analog_downsampled.dat")
                    continue
                analog_path = analog_candidates[0]
                timestamp_path = rec_dir / "MUA" / "allData" / "timestamp_s.mat"
                if not timestamp_path.exists():
                    print(f"Skip {recording_name}: missing timestamp_s.mat")
                    continue
                rec_rows.append(
                    {
                        "recording_name": recording_name,
                        "group_name": row["group_name"],
                        "analog_path": str(analog_path),
                        "timestamp_path": str(timestamp_path),
                        "good_channels": list(range(N_CHANNELS_OUT)),
                        "noisy_channels": [],
                    }
                )
                stim_rows.append(self._load_stim_df(recording_name, timestamp_path))
        return pd.DataFrame(rec_rows), (pd.concat(stim_rows, ignore_index=True) if stim_rows else pd.DataFrame())

    def _align_trials(self, firing_hz: np.ndarray, spike_times_abs: np.ndarray, stim_df: pd.DataFrame):
        pre_s = PRE_MS / 1000.0
        post_s = POST_MS / 1000.0
        n_time = len(self.time_ms)
        trials = []
        trial_spikes = []  # list of rel_times arrays per trial
        firing_tensor = np.zeros((N_CHANNELS_OUT, 0, n_time), dtype=np.float32)

        for _, row in stim_df.iterrows():
            onset = row["onset_times"]
            start_bin = int((onset - pre_s) / BIN_SIZE)
            end_bin = start_bin + n_time
            if start_bin < 0 or end_bin > firing_hz.shape[0]:
                continue
            fr_slice = firing_hz[start_bin:end_bin]
            fr_slice = fr_slice[np.newaxis, :]  # 1 x time
            firing_tensor = np.concatenate([firing_tensor, fr_slice[:, np.newaxis, :]], axis=1)

            mask = (spike_times_abs >= onset - pre_s) & (spike_times_abs <= onset + post_s)
            rel_times = spike_times_abs[mask] - onset
            trial_spikes.append(rel_times)

            trials.append(
                {
                    "trial_idx": len(trials),
                    "onset_s": onset,
                    "offset_s": row["offset_times"],
                    "stim_id": row["stimulation_ids"],
                }
            )

        spikes_rel = np.empty((N_CHANNELS_OUT, len(trial_spikes)), dtype=object)
        for i, rel_times in enumerate(trial_spikes):
            spikes_rel[0, i] = rel_times

        return firing_tensor, spikes_rel, pd.DataFrame(trials)

    def _save_recording_cache(self, rec_meta: dict, stim_df: pd.DataFrame):
        rec_name = rec_meta["recording_name"]
        rec_dir = self.output_base / rec_meta["group_name"] / rec_name
        if rec_dir.exists():
            print(f"Skip {rec_name}: output already exists at {rec_dir}")
            return
        rec_dir.mkdir(parents=True, exist_ok=False)

        firing_hz, spike_times_abs = self._compute_firing_and_spikes(Path(rec_meta["analog_path"]))
        firing_tensor, spikes_rel, trials_df = self._align_trials(firing_hz, spike_times_abs, stim_df)

        meta = {
            "recording_name": rec_name,
            "group_name": rec_meta["group_name"],
            "sampling_rate": SAMPLING_RATE,
            "bin_size": BIN_SIZE,
            "good_channels": rec_meta["good_channels"],
            "noisy_channels": rec_meta["noisy_channels"],
            "n_trials": firing_tensor.shape[1],
            "n_channels": firing_tensor.shape[0],
            "n_timebins": firing_tensor.shape[2],
            "window_ms_start": -PRE_MS,
            "window_ms_end": POST_MS,
        }

        mean_sem = {}
        for stim_id in trials_df["stim_id"].unique():
            sel = trials_df[trials_df["stim_id"] == stim_id].index.to_numpy()
            if sel.size == 0:
                continue
            fr_sel = firing_tensor[:, sel, :]
            mean_sem[f"stim_{stim_id}_mean"] = fr_sel.mean(axis=1)
            mean_sem[f"stim_{stim_id}_sem"] = fr_sel.std(axis=1) / np.sqrt(fr_sel.shape[1])

        pd.DataFrame([meta]).to_parquet(rec_dir / "meta.parquet", index=False)
        trials_df.to_parquet(rec_dir / "trials.parquet", index=False)
        np.savez(rec_dir / "firing_tensor.npz", firing_hz=firing_tensor, time_ms=self.time_ms)
        np.savez(rec_dir / "spikes_relative.npz", spikes_rel=spikes_rel, allow_pickle=True)
        np.savez(rec_dir / "psth_mean_sem.npz", **mean_sem)
        print(f"Saved cache for {rec_name} -> {rec_dir}")

    def run(self):
        if self.output_base.exists():
            raise SystemExit(f"Output base {self.output_base} already exists. Aborting to avoid overwrite.")
        recording_df, stim_df = self._gather_recordings()
        if recording_df.empty:
            raise SystemExit("No recordings found with required inputs.")
        for _, rec_row in recording_df.iterrows():
            rec_stim = stim_df[stim_df["recording_name"] == rec_row["recording_name"]]
            self._save_recording_cache(rec_row.to_dict(), rec_stim)


def main():
    builder = SpikeHistoryCacheBuilder()
    builder.run()


if __name__ == "__main__":
    main()
