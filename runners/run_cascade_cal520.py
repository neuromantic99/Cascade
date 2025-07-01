from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))

import os, sys
from scipy.ndimage import uniform_filter1d

if "Demo scripts" in os.getcwd():
    sys.path.append(os.path.abspath(".."))  # add parent directory to path for imports
    os.chdir("..")  # change to main directory
print("Current working directory: {}".format(os.getcwd()))


import numpy as np
import ruamel.yaml as yaml

yaml = yaml.YAML(typ="rt")

from cascade2p import cascade  # local folder
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution

from BaselineRemoval import BaselineRemoval


"""
Define function to load dF/F traces from disk
"""

SERVER_PATH = Path("/home/james/mnt/Rikesh/CD7")


def compute_dff(f: np.ndarray) -> np.ndarray:
    # baseline = np.percentile(f, 10, axis=1, keepdims=True)
    baseline = np.expand_dims(np.mean(f, 1), 1)
    return (f - baseline) / baseline


def compute_dff_with_rolling_mean(f: np.ndarray, N: int) -> np.ndarray:
    """
    Compute ΔF/F using a rolling mean for the fluorescence matrix.

    Parameters:
        f (np.ndarray): Fluorescence matrix (n_cells x n_frames).
        N (int): Window size for the rolling mean.

    Returns:
        np.ndarray: ΔF/F matrix with the same shape as `f`.
    """
    # Compute the rolling mean along the time axis (axis=1)
    flu_mean = uniform_filter1d(f, size=N, axis=1, mode="reflect")
    # Compute ΔF/F
    return (f - flu_mean) / flu_mean


def subtract_neuropil(f_raw: np.ndarray, f_neu: np.ndarray) -> np.ndarray:
    return f_raw - f_neu * 0.7


def load_neurons_x_time(folder: Path) -> np.ndarray:
    """Custom method to load data as 2d array with shape (neurons, nr_timepoints)"""

    s2p_path = folder / "suite2p" / "plane0"

    f_raw = np.load(s2p_path / "F.npy")
    print(f'f-raw shape {f_raw.shape}')
    f_neu = np.load(s2p_path / "Fneu.npy")

    # mean_cells = np.mean(f_raw, axis=0, keepdims=True)
    # f_norm = f_raw / mean_cells

    # dff = compute_dff(f_raw)   

    f_norm = []
    for cell in f_raw:
        baseobj = BaselineRemoval(cell)
        cell_baselined = baseobj.ZhangFit()
        f_norm.append(cell_baselined)
    f_norm = np.array(f_norm)
    # make positive
    f_norm = f_norm - f_norm.min(axis=1, keepdims=True) 
    dff = compute_dff(f_norm)

    return dff


def run_cascade(folder: Path) -> None:
    frame_rate = 30  # in Hz
    print("Loading traces from folder:", folder.name)
    traces = load_neurons_x_time(folder)
    print("Number of neurons in dataset:", traces.shape[0])
    print("Number of timepoints in dataset:", traces.shape[1])

    # cascade.download_model("update_models", verbose=1)
    model_name = "Global_EXC_30Hz_smoothing50ms_high_noise"
    # cascade.download_model(model_name, verbose=1)
    spike_prob = cascade.predict(model_name, traces)

    np.save(
        folder / "suite2p" / "plane0" / "cascade_results.npy",
        spike_prob,
    )

    np.save(
        folder / "suite2p" / "plane0" / "cascade_preprocessed.npy",
        traces,
    )


if __name__ == "__main__":
    umbrella = SERVER_PATH / "24-11-20 - Tau KO neurons" / "Cal520" / "Tau KO"
    folders = [f for f in umbrella.iterdir() if f.is_dir()]
    redo = True  

    for folder in folders:

        save_path = folder / "suite2p" / "plane0" / "cascade_results.npy"
        if save_path.exists() and not redo:
            print(f"already done {folder.name}, skipping")
            continue
        try:
            run_cascade(folder)
        except Exception as e:
            print(f"Error occured for: {folder.name}. Error: {e}")
