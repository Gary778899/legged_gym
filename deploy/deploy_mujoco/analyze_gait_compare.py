from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MIRROR_PAIRS = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]
EPS = 1e-8
WARMUP_SKIP_SEC = 0.5
CSV_ROOT_DIR = Path(__file__).resolve().parents[2] / "csv"


def parse_mirror_pairs(text: str):
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left_str, right_str = item.split(":")
        pairs.append((int(left_str), int(right_str)))
    if not pairs:
        raise ValueError("No valid mirror pairs were provided")
    return pairs


def load_and_select_window(csv_path: Path, window_sec: float, window_steps: int | None):
    dataframe = pd.read_csv(csv_path)
    if "time" in dataframe.columns:
        warmup_start = float(dataframe["time"].iloc[0]) + WARMUP_SKIP_SEC
        dataframe = dataframe[dataframe["time"] >= warmup_start].copy()

    if "time" in dataframe.columns and window_sec is not None and window_sec > 0:
        max_time = float(dataframe["time"].iloc[-1])
        min_time = max_time - float(window_sec)
        selected = dataframe[dataframe["time"] >= min_time].copy()
    elif window_steps is not None and window_steps > 0:
        selected = dataframe.tail(window_steps).copy()
    else:
        selected = dataframe.copy()

    selected = selected.reset_index(drop=True)
    if len(selected) < 20:
        raise ValueError(
            f"Selected window from {csv_path} has too few rows ({len(selected)}). "
            "Increase simulation duration or reduce window length."
        )
    return selected


def get_joint_series(dataframe: pd.DataFrame, prefix: str, joint_idx: int):
    column = f"{prefix}_joint_{joint_idx}"
    if column not in dataframe.columns:
        raise KeyError(f"Missing expected column: {column}")
    return dataframe[column].to_numpy(dtype=np.float64)


def zscore_pair(left: np.ndarray, right: np.ndarray):
    stacked = np.concatenate([left, right])
    mean = np.mean(stacked)
    std = np.std(stacked)
    if std < EPS:
        std = 1.0
    return (left - mean) / std, (right - mean) / std


def dtw_distance_exact(sequence_a: np.ndarray, sequence_b: np.ndarray, radius: int | None = None):
    n = len(sequence_a)
    m = len(sequence_b)
    if radius is None:
        radius = max(n, m)

    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(m, i + radius)
        a_i = sequence_a[i - 1]
        for j in range(j_start, j_end + 1):
            dist = np.linalg.norm(a_i - sequence_b[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(cost[n, m]) / float(n + m)


def dtw_distance(sequence_a: np.ndarray, sequence_b: np.ndarray, radius: int | None = None):
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean

        distance, _ = fastdtw(sequence_a, sequence_b, dist=euclidean)
        return float(distance) / float(len(sequence_a) + len(sequence_b))
    except Exception:
        return dtw_distance_exact(sequence_a, sequence_b, radius=radius)


def downsample(array: np.ndarray, step: int):
    if step <= 1:
        return array
    return array[::step]


def estimate_dominant_frequency_hz(signal: np.ndarray, time_values: np.ndarray | None):
    if time_values is None or len(signal) < 4 or len(signal) != len(time_values):
        return np.nan

    dt_values = np.diff(time_values)
    dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0)]
    if len(dt_values) == 0:
        return np.nan

    dt = float(np.median(dt_values))
    if not np.isfinite(dt) or dt <= 0:
        return np.nan

    centered_signal = signal - np.mean(signal)
    if np.std(centered_signal) < EPS:
        return np.nan

    spectrum = np.fft.rfft(centered_signal)
    freqs = np.fft.rfftfreq(len(centered_signal), d=dt)
    valid = freqs > 0
    if not np.any(valid):
        return np.nan

    valid_freqs = freqs[valid]
    magnitudes = np.abs(spectrum[valid])
    if len(magnitudes) == 0:
        return np.nan

    return float(valid_freqs[np.argmax(magnitudes)])


def summarize_run(
    dataframe: pd.DataFrame,
    mirror_pairs: list[tuple[int, int]],
    dtw_radius: int,
    downsample_step: int,
):
    pair_rows = []
    dtw_values = []
    amp_mismatch_values = []
    mean_asymmetry_values = []

    for left_idx, right_idx in mirror_pairs:
        qpos_left_raw = get_joint_series(dataframe, "qpos", left_idx)
        qvel_left = get_joint_series(dataframe, "qvel", left_idx)
        qpos_right_raw = get_joint_series(dataframe, "qpos", right_idx)
        qvel_right = get_joint_series(dataframe, "qvel", right_idx)

        mean_asymmetry = float(abs((qpos_left_raw - qpos_right_raw).mean()))

        qpos_left, qpos_right = zscore_pair(qpos_left_raw, qpos_right_raw)

        qvel_left, qvel_right = zscore_pair(qvel_left, qvel_right)

        trajectory_left = np.column_stack([qpos_left, qvel_left])
        trajectory_right = np.column_stack([qpos_right, qvel_right])

        trajectory_left = downsample(trajectory_left, downsample_step)
        trajectory_right = downsample(trajectory_right, downsample_step)

        dtw_val = dtw_distance(trajectory_left, trajectory_right, radius=dtw_radius)

        amp_qpos = abs(np.std(qpos_left) - np.std(qpos_right)) / (0.5 * (np.std(qpos_left) + np.std(qpos_right)) + EPS)
        amp_qvel = abs(np.std(qvel_left) - np.std(qvel_right)) / (0.5 * (np.std(qvel_left) + np.std(qvel_right)) + EPS)
        amp_mismatch = 0.5 * (amp_qpos + amp_qvel)

        dtw_values.append(dtw_val)
        amp_mismatch_values.append(amp_mismatch)
        mean_asymmetry_values.append(mean_asymmetry)

        pair_rows.append(
            {
                "pair": f"{left_idx}:{right_idx}",
                "dtw_symmetry": dtw_val,
                "amp_mismatch": amp_mismatch,
                "mean_asymmetry": mean_asymmetry,
            }
        )

    time_values = (
        dataframe["time"].to_numpy(dtype=np.float64) if "time" in dataframe.columns else None
    )
    hip_pitch_series = get_joint_series(dataframe, "qpos", 0)
    dominant_freq_hz_hip_pitch = estimate_dominant_frequency_hz(hip_pitch_series, time_values)
    gait_period_s_hip_pitch = (
        float(1.0 / dominant_freq_hz_hip_pitch)
        if np.isfinite(dominant_freq_hz_hip_pitch) and dominant_freq_hz_hip_pitch > 0
        else np.nan
    )

    ctrl_cols = [column for column in dataframe.columns if column.startswith("ctrl_joint_")]
    ctrl_values = dataframe[ctrl_cols].to_numpy(dtype=np.float64) if ctrl_cols else np.empty((len(dataframe), 0))

    if ctrl_values.shape[1] > 0 and len(ctrl_values) > 1:
        action_rate_rms = float(np.sqrt(np.mean(np.diff(ctrl_values, axis=0) ** 2)))
        control_effort_mean = float(np.mean(np.abs(ctrl_values)))
    else:
        action_rate_rms = np.nan
        control_effort_mean = np.nan

    metrics = {
        "symmetry_dtw_mean": float(np.mean(dtw_values)),
        "symmetry_dtw_knee": float(next(row["dtw_symmetry"] for row in pair_rows if row["pair"] == "3:9")),
        "mean_asymmetry_mean": float(np.mean(mean_asymmetry_values)),
        "amp_mismatch_mean": float(np.mean(amp_mismatch_values)),
        "dominant_freq_hz_hip_pitch": dominant_freq_hz_hip_pitch,
        "gait_period_s_hip_pitch": gait_period_s_hip_pitch,
        "action_rate_rms": action_rate_rms,
        "control_effort_mean_abs": control_effort_mean,
        "num_samples": int(len(dataframe)),
    }

    return metrics, pd.DataFrame(pair_rows)


def print_metrics_table(baseline_metrics: dict, test_metrics: dict):
    keys = [
        "symmetry_dtw_mean",
        "symmetry_dtw_knee",
        "mean_asymmetry_mean",
        "amp_mismatch_mean",
        "dominant_freq_hz_hip_pitch",
        "gait_period_s_hip_pitch",
        "action_rate_rms",
        "control_effort_mean_abs",
        "num_samples",
    ]

    print("\n=== Baseline vs Test Metrics (lower is better except num_samples) ===")
    print(f"{'metric':28s} {'baseline':>14s} {'test':>14s} {'delta(test-baseline)':>22s}")
    for key in keys:
        b_val = baseline_metrics[key]
        t_val = test_metrics[key]
        delta = t_val - b_val
        if isinstance(b_val, (int, np.integer)):
            print(f"{key:28s} {b_val:14d} {int(t_val):14d} {int(delta):22d}")
        else:
            print(f"{key:28s} {b_val:14.6f} {t_val:14.6f} {delta:22.6f}")


def save_phase_portrait_plot(
    baseline_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_path: Path,
    left_knee_idx: int,
    right_knee_idx: int,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Skip phase portrait plot: matplotlib unavailable ({exc})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    for axis, dataframe, title in [
        (axes[0], baseline_df, "Baseline"),
        (axes[1], test_df, "Test"),
    ]:
        qpos_left = get_joint_series(dataframe, "qpos", left_knee_idx)
        qvel_left = get_joint_series(dataframe, "qvel", left_knee_idx)
        qpos_right = get_joint_series(dataframe, "qpos", right_knee_idx)
        qvel_right = get_joint_series(dataframe, "qvel", right_knee_idx)

        axis.plot(qpos_left, qvel_left, label="Left knee", linewidth=1.6, color="#1f77b4")
        axis.plot(qpos_right, qvel_right, label="Right knee", linewidth=1.6, linestyle="--", color="#ff7f0e")
        axis.set_title(f"{title} knee phase portrait")
        axis.set_xlabel("Joint position (rad)")
        axis.set_ylabel("Joint velocity (rad/s)")
        axis.grid(True, linestyle=":", alpha=0.6)
        axis.legend(loc="best")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_pairwise_bar_plot(baseline_pairs: pd.DataFrame, test_pairs: pd.DataFrame, out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Skip pairwise DTW plot: matplotlib unavailable ({exc})")
        return

    merged = baseline_pairs.merge(test_pairs, on="pair", suffixes=("_baseline", "_test"))
    x = np.arange(len(merged))
    width = 0.38

    fig, axis = plt.subplots(figsize=(10, 4.8), dpi=200)
    axis.bar(x - width / 2, merged["dtw_symmetry_baseline"], width, label="Baseline", color="#4c78a8")
    axis.bar(x + width / 2, merged["dtw_symmetry_test"], width, label="Test", color="#f58518")

    axis.set_xticks(x)
    axis.set_xticklabels(merged["pair"].tolist())
    axis.set_xlabel("Mirror joint pair (left:right)")
    axis.set_ylabel("DTW symmetry distance (lower better)")
    axis.set_title("Per-joint-pair symmetry comparison")
    axis.grid(True, axis="y", linestyle=":", alpha=0.6)
    axis.legend(loc="best")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare gait quality metrics from two MuJoCo CSV logs")
    parser.add_argument("--baseline_csv", required=True, help="Path to baseline CSV")
    parser.add_argument("--test_csv", required=True, help="Path to test CSV")
    parser.add_argument("--output_dir", default="gait_compare_output", help="Output subdirectory name under csv/")
    parser.add_argument(
        "--mirror_pairs",
        default="0:6,1:7,2:8,3:9,4:10,5:11",
        help="Comma-separated mirror pairs like 0:6,1:7,2:8",
    )
    parser.add_argument("--window_sec", type=float, default=20.0, help="Analyze only the last N seconds if time exists")
    parser.add_argument(
        "--window_steps",
        type=int,
        default=None,
        help="Fallback number of tail steps when time column is unavailable",
    )
    parser.add_argument("--downsample", type=int, default=4, help="Downsample factor before DTW")
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip plot generation and export only numeric metrics",
    )
    parser.add_argument(
        "--dtw_radius",
        type=int,
        default=200,
        help="Sakoe-Chiba radius for exact DTW fallback (ignored by fastdtw)",
    )
    args = parser.parse_args()

    baseline_csv = Path(args.baseline_csv).expanduser().resolve()
    test_csv = Path(args.test_csv).expanduser().resolve()
    output_subdir = Path(args.output_dir).name
    output_dir = (CSV_ROOT_DIR / output_subdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mirror_pairs = parse_mirror_pairs(args.mirror_pairs)

    baseline_df = load_and_select_window(baseline_csv, args.window_sec, args.window_steps)
    test_df = load_and_select_window(test_csv, args.window_sec, args.window_steps)

    baseline_metrics, baseline_pairs = summarize_run(
        baseline_df,
        mirror_pairs,
        dtw_radius=args.dtw_radius,
        downsample_step=args.downsample,
    )
    test_metrics, test_pairs = summarize_run(
        test_df,
        mirror_pairs,
        dtw_radius=args.dtw_radius,
        downsample_step=args.downsample,
    )

    print_metrics_table(baseline_metrics, test_metrics)

    summary = {
        "baseline_csv": str(baseline_csv),
        "test_csv": str(test_csv),
        "settings": {
            "window_sec": args.window_sec,
            "window_steps": args.window_steps,
            "downsample": args.downsample,
            "dtw_radius": args.dtw_radius,
            "mirror_pairs": mirror_pairs,
        },
        "baseline_metrics": baseline_metrics,
        "test_metrics": test_metrics,
    }

    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    baseline_pairs.to_csv(output_dir / "pairwise_metrics_baseline.csv", index=False)
    test_pairs.to_csv(output_dir / "pairwise_metrics_test.csv", index=False)

    if not args.skip_plots:
        save_phase_portrait_plot(
            baseline_df,
            test_df,
            output_dir / "knee_phase_portrait_comparison.png",
            left_knee_idx=3,
            right_knee_idx=9,
        )
        save_pairwise_bar_plot(
            baseline_pairs,
            test_pairs,
            output_dir / "pairwise_dtw_comparison.png",
        )

    print("\nSaved outputs:")
    print(f"- {output_dir / 'summary_metrics.json'}")
    print(f"- {output_dir / 'pairwise_metrics_baseline.csv'}")
    print(f"- {output_dir / 'pairwise_metrics_test.csv'}")
    if not args.skip_plots:
        print(f"- {output_dir / 'knee_phase_portrait_comparison.png'}")
        print(f"- {output_dir / 'pairwise_dtw_comparison.png'}")


if __name__ == "__main__":
    main()
