from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MIRROR_PAIRS = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]
EPS = 1e-8
WARMUP_SKIP_SEC = 0.5
STARTUP_START_SEC = 0.0
STARTUP_END_SEC = 2.0
STARTUP_FALLBACK_STEPS = 400
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


def load_csv(csv_path: Path):
    dataframe = pd.read_csv(csv_path)
    if len(dataframe) < 20:
        raise ValueError(
            f"CSV {csv_path} has too few rows ({len(dataframe)}). "
            "Run a longer simulation before analysis."
        )
    return dataframe


def select_steady_state_window(
    dataframe: pd.DataFrame,
    window_sec: float,
    window_steps: int | None,
    warmup_skip_sec: float,
):
    if "time" in dataframe.columns:
        warmup_start = float(dataframe["time"].iloc[0]) + warmup_skip_sec
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
            f"Selected steady-state window has too few rows ({len(selected)}). "
            "Increase simulation duration or reduce window length."
        )
    return selected


def select_startup_window(
    dataframe: pd.DataFrame,
    startup_start_sec: float,
    startup_end_sec: float,
    startup_fallback_steps: int,
):
    if startup_end_sec <= startup_start_sec:
        raise ValueError("startup_end_sec must be greater than startup_start_sec")

    if "time" in dataframe.columns:
        t0 = float(dataframe["time"].iloc[0])
        start_time = t0 + startup_start_sec
        end_time = t0 + startup_end_sec
        startup = dataframe[
            (dataframe["time"] >= start_time) & (dataframe["time"] <= end_time)
        ].copy()
    else:
        startup = dataframe.head(startup_fallback_steps).copy()

    startup = startup.reset_index(drop=True)
    if len(startup) < 20:
        raise ValueError(
            f"Selected startup window has too few rows ({len(startup)}). "
            "Increase simulation duration or widen startup window."
        )
    return startup


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
        fastdtw = importlib.import_module("fastdtw").fastdtw
        euclidean = importlib.import_module("scipy.spatial.distance").euclidean

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


def estimate_sample_rate_hz(time_values: np.ndarray | None):
    if time_values is None or len(time_values) < 2:
        return np.nan

    dt_values = np.diff(time_values)
    dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0)]
    if len(dt_values) == 0:
        return np.nan

    dt = float(np.median(dt_values))
    if not np.isfinite(dt) or dt <= 0:
        return np.nan
    return float(1.0 / dt)


def estimate_ctrl_metrics(dataframe: pd.DataFrame):
    ctrl_cols = [column for column in dataframe.columns if column.startswith("ctrl_joint_")]
    ctrl_values = dataframe[ctrl_cols].to_numpy(dtype=np.float64) if ctrl_cols else np.empty((len(dataframe), 0))

    if ctrl_values.shape[1] > 0 and len(ctrl_values) > 1:
        deltas = np.diff(ctrl_values, axis=0)
        action_rate_rms = float(np.sqrt(np.mean(deltas ** 2)))
        control_effort_mean = float(np.mean(np.abs(ctrl_values)))
        max_abs_delta_ctrl = float(np.max(np.abs(deltas)))
    else:
        action_rate_rms = np.nan
        control_effort_mean = np.nan
        max_abs_delta_ctrl = np.nan

    return {
        "action_rate_rms": action_rate_rms,
        "control_effort_mean_abs": control_effort_mean,
        "max_abs_delta_ctrl": max_abs_delta_ctrl,
    }


def summarize_height_stats(dataframe: pd.DataFrame, base_height_target: float):
    if "base_qpos_2" not in dataframe.columns:
        raise KeyError("Missing expected column: base_qpos_2")

    heights = dataframe["base_qpos_2"].to_numpy(dtype=np.float64)
    abs_err = np.abs(heights - base_height_target)
    return {
        "base_height_mean": float(np.mean(heights)),
        "base_height_std": float(np.std(heights)),
        "base_height_min": float(np.min(heights)),
        "base_height_max": float(np.max(heights)),
        "base_height_pct_below_target": float(np.mean(heights < base_height_target) * 100.0),
        "base_height_abs_err_mean": float(np.mean(abs_err)),
        "base_height_abs_err_std": float(np.std(abs_err)),
        "base_height_abs_err_max": float(np.max(abs_err)),
    }


def summarize_run(
    dataframe: pd.DataFrame,
    mirror_pairs: list[tuple[int, int]],
    dtw_radius: int,
    downsample_step: int,
):
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

    time_values = dataframe["time"].to_numpy(dtype=np.float64) if "time" in dataframe.columns else None
    hip_pitch_series = get_joint_series(dataframe, "qpos", 0)
    dominant_freq_hz_hip_pitch = estimate_dominant_frequency_hz(hip_pitch_series, time_values)
    gait_period_s_hip_pitch = (
        float(1.0 / dominant_freq_hz_hip_pitch)
        if np.isfinite(dominant_freq_hz_hip_pitch) and dominant_freq_hz_hip_pitch > 0
        else np.nan
    )

    ctrl_metrics = estimate_ctrl_metrics(dataframe)

    metrics = {
        "symmetry_dtw_mean": float(np.mean(dtw_values)),
        "symmetry_dtw_knee": float(dtw_values[mirror_pairs.index((3, 9))]) if (3, 9) in mirror_pairs else np.nan,
        "mean_asymmetry_mean": float(np.mean(mean_asymmetry_values)),
        "amp_mismatch_mean": float(np.mean(amp_mismatch_values)),
        "dominant_freq_hz_hip_pitch": dominant_freq_hz_hip_pitch,
        "gait_period_s_hip_pitch": gait_period_s_hip_pitch,
        "action_rate_rms": ctrl_metrics["action_rate_rms"],
        "control_effort_mean_abs": ctrl_metrics["control_effort_mean_abs"],
        "num_samples": int(len(dataframe)),
    }

    return metrics


def summarize_startup(
    startup_df: pd.DataFrame,
    mirror_pairs: list[tuple[int, int]],
    base_height_target: float,
):
    if "base_qpos_2" not in startup_df.columns:
        raise KeyError("Missing expected column: base_qpos_2")

    startup_heights = startup_df["base_qpos_2"].to_numpy(dtype=np.float64)
    startup_abs_err = np.abs(startup_heights - base_height_target)

    startup_asymmetry = []
    for left_idx, right_idx in mirror_pairs:
        qpos_left = get_joint_series(startup_df, "qpos", left_idx)
        qpos_right = get_joint_series(startup_df, "qpos", right_idx)
        startup_asymmetry.append(float(np.mean(np.abs(qpos_left - qpos_right))))

    startup_ctrl_metrics = estimate_ctrl_metrics(startup_df)

    return {
        "startup_num_samples": int(len(startup_df)),
        "startup_min_base_qpos_2": float(np.min(startup_heights)),
        "startup_pct_below_target": float(np.mean(startup_heights < base_height_target) * 100.0),
        "startup_height_abs_err_mean": float(np.mean(startup_abs_err)),
        "startup_height_abs_err_std": float(np.std(startup_abs_err)),
        "startup_height_abs_err_max": float(np.max(startup_abs_err)),
        "startup_action_rate_rms": startup_ctrl_metrics["action_rate_rms"],
        "startup_max_abs_delta_ctrl": startup_ctrl_metrics["max_abs_delta_ctrl"],
        "startup_mean_abs_lr_asymmetry": float(np.mean(startup_asymmetry)),
    }


def evaluate_logging_frequency(
    full_df: pd.DataFrame,
    dominant_freq_hz: float,
):
    time_values = full_df["time"].to_numpy(dtype=np.float64) if "time" in full_df.columns else None
    sample_rate_hz = estimate_sample_rate_hz(time_values)

    if np.isfinite(dominant_freq_hz) and dominant_freq_hz > 0:
        required_hz = max(20.0, 8.0 * dominant_freq_hz)
    else:
        required_hz = 20.0

    if np.isfinite(sample_rate_hz) and sample_rate_hz > 0:
        safety_ratio = float(sample_rate_hz / required_hz)
        if safety_ratio >= 2.0:
            recommendation = (
                "Current logging frequency is conservative and can be downsampled in analysis "
                "without losing core gait/startup metrics. Keep raw logging unchanged for traceability."
            )
            max_downsample = int(np.floor(safety_ratio))
        else:
            recommendation = (
                "Current logging frequency is near the recommended floor for robust startup/control metrics; "
                "avoid reducing logger frequency."
            )
            max_downsample = 1
    else:
        safety_ratio = np.nan
        recommendation = "Cannot estimate logging frequency because time column is missing or invalid."
        max_downsample = 1

    return {
        "estimated_log_hz": sample_rate_hz,
        "recommended_min_hz": float(required_hz),
        "sampling_safety_ratio": safety_ratio,
        "suggested_max_analysis_downsample": int(max_downsample),
        "logging_recommendation": recommendation,
    }


def _format_metric(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    if isinstance(value, str):
        return value
    return f"{float(value):.6f}"


def write_summary_txt(
    out_path: Path,
    input_csv: Path,
    settings: dict,
    startup_metrics: dict,
    steady_metrics: dict,
    height_metrics: dict,
    logging_eval: dict,
):
    lines = []
    lines.append("Round-0 MuJoCo Gait Quantification Summary")
    lines.append("=" * 44)
    lines.append("")
    lines.append("Input")
    lines.append("-----")
    lines.append(f"csv_file: {input_csv}")
    lines.append("")
    lines.append("Settings")
    lines.append("--------")
    for key, value in settings.items():
        lines.append(f"{key}: {_format_metric(value)}")
    lines.append("")
    lines.append("Startup Metrics (0.0-2.0s)")
    lines.append("-------------------------")
    for key, value in startup_metrics.items():
        lines.append(f"{key}: {_format_metric(value)}")
    lines.append("")
    lines.append("Steady-State Metrics")
    lines.append("--------------------")
    for key, value in steady_metrics.items():
        lines.append(f"{key}: {_format_metric(value)}")
    lines.append("")
    lines.append("Base Height Statistics")
    lines.append("----------------------")
    for key, value in height_metrics.items():
        lines.append(f"{key}: {_format_metric(value)}")
    lines.append("")
    lines.append("Logging Frequency Evaluation")
    lines.append("----------------------------")
    for key, value in logging_eval.items():
        lines.append(f"{key}: {_format_metric(value)}")
    lines.append("")
    lines.append("Metric direction hints:")
    lines.append("- Lower is generally better: symmetry_dtw_*, mean_asymmetry_*, amp_mismatch_*, action_rate_rms, control_effort_mean_abs, *_abs_err_*, *_pct_below_target")
    lines.append("- For dominant_freq_hz_hip_pitch and gait_period_s_hip_pitch: compare against your desired cadence target.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Quantify startup and gait metrics from one MuJoCo logger CSV")
    parser.add_argument("--log_csv", required=True, help="Path to logger CSV generated by deploy_mujoco.py")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to the input CSV directory.",
    )
    parser.add_argument(
        "--mirror_pairs",
        default="0:6,1:7,2:8,3:9,4:10,5:11",
        help="Comma-separated mirror pairs like 0:6,1:7,2:8",
    )
    parser.add_argument("--window_sec", type=float, default=20.0, help="Analyze only the last N seconds for steady-state metrics if time exists")
    parser.add_argument(
        "--window_steps",
        type=int,
        default=None,
        help="Fallback number of tail steps when time column is unavailable",
    )
    parser.add_argument("--downsample", type=int, default=4, help="Downsample factor before DTW")
    parser.add_argument("--base_height_target", type=float, default=0.65, help="Target base height used in startup/height error metrics")
    parser.add_argument("--warmup_skip_sec", type=float, default=WARMUP_SKIP_SEC, help="Warmup skip for steady-state analysis")
    parser.add_argument("--startup_start_sec", type=float, default=STARTUP_START_SEC, help="Startup analysis start time in seconds from first sample")
    parser.add_argument("--startup_end_sec", type=float, default=STARTUP_END_SEC, help="Startup analysis end time in seconds from first sample")
    parser.add_argument(
        "--startup_fallback_steps",
        type=int,
        default=STARTUP_FALLBACK_STEPS,
        help="Fallback startup steps when time column is unavailable",
    )
    parser.add_argument(
        "--dtw_radius",
        type=int,
        default=200,
        help="Sakoe-Chiba radius for exact DTW fallback (ignored by fastdtw)",
    )
    args = parser.parse_args()

    log_csv = Path(args.log_csv).expanduser().resolve()
    if args.output_dir is None:
        output_dir = log_csv.parent
    else:
        output_subdir = Path(args.output_dir).name
        output_dir = (CSV_ROOT_DIR / output_subdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mirror_pairs = parse_mirror_pairs(args.mirror_pairs)

    full_df = load_csv(log_csv)
    startup_df = select_startup_window(
        full_df,
        startup_start_sec=args.startup_start_sec,
        startup_end_sec=args.startup_end_sec,
        startup_fallback_steps=args.startup_fallback_steps,
    )
    steady_df = select_steady_state_window(
        full_df,
        args.window_sec,
        args.window_steps,
        warmup_skip_sec=args.warmup_skip_sec,
    )

    startup_metrics = summarize_startup(
        startup_df,
        mirror_pairs,
        base_height_target=args.base_height_target,
    )
    steady_metrics = summarize_run(
        steady_df,
        mirror_pairs,
        dtw_radius=args.dtw_radius,
        downsample_step=args.downsample,
    )
    height_metrics = summarize_height_stats(steady_df, base_height_target=args.base_height_target)
    logging_eval = evaluate_logging_frequency(
        full_df,
        dominant_freq_hz=steady_metrics["dominant_freq_hz_hip_pitch"],
    )

    settings = {
        "window_sec": args.window_sec,
        "window_steps": args.window_steps if args.window_steps is not None else "none",
        "downsample": args.downsample,
        "dtw_radius": args.dtw_radius,
        "mirror_pairs": args.mirror_pairs,
        "base_height_target": args.base_height_target,
        "warmup_skip_sec": args.warmup_skip_sec,
        "startup_start_sec": args.startup_start_sec,
        "startup_end_sec": args.startup_end_sec,
        "startup_fallback_steps": args.startup_fallback_steps,
    }

    summary_txt = output_dir / f"{log_csv.stem}_summary.txt"
    write_summary_txt(
        out_path=summary_txt,
        input_csv=log_csv,
        settings=settings,
        startup_metrics=startup_metrics,
        steady_metrics=steady_metrics,
        height_metrics=height_metrics,
        logging_eval=logging_eval,
    )

    print("\nSaved outputs:")
    print(f"- {summary_txt}")


if __name__ == "__main__":
    main()
