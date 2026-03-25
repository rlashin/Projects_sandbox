import numpy as np
from pathlib import Path
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
import matplotlib.pyplot as plt
from neuropy.analyses.oscillations import _detect_freq_band_epochs
from neuropy.core import Signal, Epoch, ProbeGroup
from neuropy.utils import signal_process

animal_dir = Path(r"D:\data\Nat\Psilocybin\Recording_Rats\Finn")
sessions = ["saline1", "psilocybin", "saline2"]
animal_name = "Finn2"
fig, ax = plt.subplots(1, 3, figsize=(11.3, 1.2))
fig.suptitle(animal_name)
ax[0].set_title("Saline1")
ax[1].set_title("Psilocybin")
ax[2].set_title("Saline2")

def detect_hvs_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(12, 18),  # HVS frequency band based on Buzcode NotchHVS
    thresh=(2, None),  # Threshold for high power detection
    edge_cutoff=1.0,  # Edge cutoff for z-scored power
    mindur=0.35,
    maxdur=4,
    mergedist=0.05,
    sigma=0.125,
    ignore_epochs: Epoch = None,
    basepath: str = r"D:\data\Nat\Psilocybin\Recording_Rats\Finn\2022_02_17_psilocybin",
):
    """
    Detect high voltage spindle (HVS) epochs.

    HVS are detected in the 12-18 Hz band,
    which is the frequency range notched out in Buzzcode when NotchHVS=true.

    Parameters
    ----------
    signal : Signal
        LFP signal object
    probegroup : ProbeGroup, optional
        Probe group for channel selection
    freq_band : tuple, optional
        Frequency band for HVS detection (default 12-18 Hz)
    thresh : tuple, optional
        Low and high threshold for detection (default (2, None))
    edge_cutoff : float, optional
        Edge cutoff for z-scored power (default 1.0)
    mindur : float, optional
        Minimum duration in seconds
    maxdur : float, optional
        Maximum duration in seconds
    mergedist : float, optional
        Merge distance for overlapping epochs
    sigma : float, optional
        Gaussian smoothing sigma
    ignore_epochs : Epoch, optional
        Epochs to ignore during detection
    basepath : str, optional
        Path to the session for logging purposes

    Returns
    -------
    Epoch
        Detected HVS epochs
    """
    print("Starting HVS detection")
    if basepath:
        print(f"Processing session: {basepath}")
    print(f"Signal sampling rate: {signal.sampling_rate}, channels: {signal.channel_id}")

    if probegroup is None:
        selected_chans = signal.channel_id
        traces = signal.traces
    else:
        # Select best channel for HVS detection (similar to spindle detection)
        channel_ids = np.concatenate(probegroup.get_connected_channels(groupby="shank")).astype("int")
        signal_slice = signal.time_slice(channel_id=channel_ids, t_start=signal.t_start,
                                       t_stop=min(signal.t_start + 3600, signal.t_stop))
        hil_stat = signal_process.hilbert_amplitude_stat(
            signal_slice.traces, freq_band=freq_band, fs=signal.sampling_rate, statistic="mean"
        )
        selected_chans = channel_ids[np.argmax(hil_stat)].reshape(-1)
        traces = signal.time_slice(channel_id=selected_chans).traces.reshape(1, -1)

    print(f"Selected channel for HVS: {selected_chans}")

    if ignore_epochs is not None:
        ignore_times = ignore_epochs.shift(-signal.t_start).as_array()
    else:
        ignore_times = None

    epochs = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        edge_cutoff=edge_cutoff,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        sigma=sigma,
        ignore_times=ignore_times,
    )
    epochs = epochs.shift(dt=signal.t_start)
    epochs.metadata = dict(channels=selected_chans, freq_band=freq_band)
    print(f"Detected {len(epochs)} HVS epochs.")
    return epochs


def plot_hvs_epochs(epochs, ax=None, color='red', alpha=0.5, label='HVS'):
    """
    Plot HVS epochs over time.

    Parameters
    ----------
    epochs : Epoch
        HVS epochs to plot
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates new figure.
    color : str, optional
        Color for the epochs
    alpha : float, optional
        Transparency
    label : str, optional
        Label for the plot

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2))

    for start, stop in epochs.as_array():
        ax.axvspan(start, stop, alpha=alpha, color=color, label=label if start == epochs.as_array()[0][0] else "")

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('HVS Presence')
    ax.set_title('High Voltage Spindles Over Time')
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    if label:
        ax.legend()

    return ax



signal = Signal(traces=np.random.randn(1, 100000), sampling_rate=__, channel_id=[0])
hvs_epochs = detect_hvs_epochs(signal)
# plot_hvs_epochs(hvs_epochs)
# plt.show()
plot_hvs_epochs(hvs_epochs)
plt.show()
