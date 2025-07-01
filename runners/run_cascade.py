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


from cascade2p.gsheets_importer import gsheet2df

"""
Define function to load dF/F traces from disk
"""

SERVER_PATH = Path("/home/james/mnt/James/Regular2p")


def compute_dff(f: np.ndarray) -> np.ndarray:
    flu_mean = np.expand_dims(np.mean(f, 1), 1)
    return (f - flu_mean) / flu_mean

def load_neurons_x_time(mouse: str, date: str) -> np.ndarray:
    """Custom method to load data as 2d array with shape (neurons, nr_timepoints)"""

    s2p_path = SERVER_PATH / date / mouse / "suite2p" / "plane0"
    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)

    f_raw = np.load(s2p_path / "F.npy")[iscell, :]
    print(f'f-raw shape {f_raw.shape}')
    f_neu = np.load(s2p_path / "Fneu.npy")[iscell, :]

    #dff = compute_dff_with_rolling_mean(subtract_neuropil(f_raw, f_neu), 30*30)
    dff = compute_dff(subtract_neuropil(f_raw, f_neu))

    # dff = dff - np.expand_dims(np.min(dff, 1), 1)
    print(f'dff shape {dff.shape}')

    return dff


def run_cascade(mouse: str, date: str) -> None:
    frame_rate = 30  # in Hz
    traces = load_neurons_x_time(mouse, date)
    print("Number of neurons in dataset:", traces.shape[0])
    print("Number of timepoints in dataset:", traces.shape[1])

    noise_levels = plot_noise_level_distribution(traces, frame_rate)
    neuron_indices = np.random.randint(traces.shape[0], size=10)

    cascade.download_model("update_models", verbose=1)

    yaml_file = open("Pretrained_models/available_models.yaml")
    model_name = "GC8s_EXC_30Hz_smoothing50ms_high_noise"
    cascade.download_model(model_name, verbose=1)
    spike_prob = cascade.predict(model_name, traces)

    np.save(
        SERVER_PATH / date / mouse / "suite2p" / "plane0" / "cascade_results_not_zeroed.npy",
        spike_prob,
    )
    np.save(
        SERVER_PATH / date / mouse / "suite2p" / "plane0" / "noise_levels_cascade_not_zeroed.npy",
        noise_levels,
    )


if __name__ == "__main__":
    metadata = gsheet2df("1NZi5kRUMJMPte7jeRmqFrYJIPbUPMDZ_Yq4lWE2ql1k", "Sheet1", 1)
    metadata = metadata[metadata["Anatomical suite2p clicked"] == "DONE"]

    for idx, row in metadata.iterrows():
        mouse = row["Mouse"]
        date = row["Date"]

        save_path = SERVER_PATH / date / mouse / "suite2p" / "plane0" / "cascade_results_not_zeroed.npy"
        if save_path.exists():
            print(f"already down {mouse}, {date}")
            continue
        try:
            run_cascade(mouse, date)
        except Exception as e:
            print(f"Error occured for: {mouse} {date}. Error: {e}")
