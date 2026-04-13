from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]

# Set the .npz path here.
# Default is the first timestamp folder's HEF_outputs file.
NPZ_PATH = BASE_DIR / "results/csv/eqcct_earthquakes_subset/20250101T092157Z_20250101T092257Z/HEF_outputs/X_probability_traces.npz"


def main() -> None:
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"NPZ file not found: {NPZ_PATH}")

    with np.load(NPZ_PATH, allow_pickle=True) as npz:
        p_probs = np.asarray(npz["p_probabilities"])
        s_probs = np.asarray(npz["s_probabilities"])

        if p_probs.ndim == 1:
            p_probs = p_probs.reshape(1, -1)
        if s_probs.ndim == 1:
            s_probs = s_probs.reshape(1, -1)

        trace_start_times = (
            np.asarray(npz["trace_start_time"], dtype=str)
            if "trace_start_time" in npz.files
            else np.array([], dtype=str)
        )
        sample_rate_hz = (
            float(np.asarray(npz["sample_rate_hz"]).reshape(-1)[0])
            if "sample_rate_hz" in npz.files
            else 100.0
        )

    n_traces = min(p_probs.shape[0], s_probs.shape[0])
    print(f"Using NPZ: {NPZ_PATH}")
    print(f"Trace count: {n_traces}, sample_rate_hz: {sample_rate_hz}")
    print("Displaying plots only (no files will be saved).")

    for idx in range(n_traces):
        p_trace = p_probs[idx]
        s_trace = s_probs[idx]
        n_samples = min(len(p_trace), len(s_trace))
        x_seconds = np.arange(n_samples, dtype=float) / sample_rate_hz

        start_time = trace_start_times[idx] if idx < len(trace_start_times) else f"trace_{idx:04d}"

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x_seconds, p_trace[:n_samples], linewidth=1.0, label="P probability")
        ax.plot(x_seconds, s_trace[:n_samples], linewidth=1.0, label="S probability")
        ax.set_xlabel("Seconds from trace start")
        ax.set_ylabel("Probability")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"Probability traces | start={start_time}")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
        fig.tight_layout()
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
