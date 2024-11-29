from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))

import os, sys

if "Demo scripts" in os.getcwd():
    sys.path.append(os.path.abspath(".."))  # add parent directory to path for imports
    os.chdir("..")  # change to main directory
print("Current working directory: {}".format(os.getcwd()))


import numpy as np
import ruamel.yaml as yaml

yaml = yaml.YAML(typ="rt")

from cascade2p import cascade  # local folder
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution

from cascade2p import gsheets_importer
1/0


"""
Define function to load dF/F traces from disk
"""

SERVER_PATH = Path("/home/james/mnt/James/Regular2p")


def compute_dff(f: np.ndarray) -> np.ndarray:
    flu_mean = np.expand_dims(np.mean(f, 1), 1)
    return (f - flu_mean) / flu_mean


def subtract_neuropil(f_raw: np.ndarray, f_neu: np.ndarray) -> np.ndarray:
    return f_raw - f_neu * 0.7


def load_neurons_x_time(mouse: str, date: str) -> np.ndarray:
    """Custom method to load data as 2d array with shape (neurons, nr_timepoints)"""

    s2p_path = SERVER_PATH / date / mouse / "suite2p" / "plane0"
    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)
    f_raw = np.load(s2p_path / "F.npy")[iscell, :]
    f_neu = np.load(s2p_path / "Fneu.npy")[iscell, :]
    dff = compute_dff(subtract_neuropil(f_raw, f_neu))
    dff = dff - np.expand_dims(np.min(dff, 1), 1)

    return dff


def run_cascade(mouse: str, date: str) -> None:
    frame_rate = 30  # in Hz
    traces = load_neurons_x_time(mouse, date)
    print("Number of neurons in dataset:", traces.shape[0])
    print("Number of timepoints in dataset:", traces.shape[1])

    noise_levels = plot_noise_level_distribution(traces, frame_rate)
    neuron_indices = np.random.randint(traces.shape[0], size=10)
    plot_dFF_traces(traces, neuron_indices, frame_rate)

    cascade.download_model("update_models", verbose=1)

    yaml_file = open("Pretrained_models/available_models.yaml")
    X = yaml.load(yaml_file)
    model_name = "GC8s_EXC_30Hz_smoothing50ms_high_noise"
    cascade.download_model(model_name, verbose=1)
    spike_prob = cascade.predict(model_name, traces)

    np.save(
        SERVER_PATH / date / mouse / "suite2p" / "plane0" / "cascade_results.npy",
        spike_prob,
    )
    np.save(
        SERVER_PATH / date / mouse / "suite2p" / "plane0" / "noise_levels_cascade.npy",
        noise_levels,
    )


if __name__ == "__main__":
    for date in ["2024-09-27", "2024-10-09", "2024-10-24", "2024-10-25"]:
        for mouse in [
            "J022",
            "J023",
            "J024",
            "J025",
            "J026",
            "J027",
            "J029",
            "J029",
        ]:
            try:
                run_cascade(mouse, date)
            except Exception as e:
                print(f"Error occured for: {mouse} {date}. Error: {e}")
