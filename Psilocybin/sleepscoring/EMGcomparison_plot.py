from pathlib import Path
import matplotlib.pyplot as plt
from Brainstates_ryan import emg_from_LFP
from neuropy.io.sleepscoremasterio import SleepScoreIO
from neuropy.plotting.epochs import plot_hypnogram
from neuropy.utils.plot_util import match_axis_lims
import seaborn as sns
import numpy as np

emg_hist_lims = {"Finn": [-0.16, 0.91]}
nbins = 40

psilocybin_dir = Path(r"D:\data\Nat\Psilocybin\Recording_Rats")
alt_dir = Path(r"D:\data\Nat\Alternation\Recording_Rats")
animal_name = "Finn"
sessions = ["alternation", "psilocybin"]
fig, ax = plt.subplots(1, 2, figsize=(11.3, 1.2))
fig.suptitle("EMG Comparison")
ax[0].set_title("Alternation vs Psilocybin")

for ids, (base_dir, session_type) in enumerate(zip([alt_dir, psilocybin_dir], ["alternation*", "psilocybin"])):
    sess_dir = sorted((base_dir / animal_name).glob(f"*_{session_type}"))[0]
    sleep = SleepScoreIO(sess_dir)
    EMG = sleep.read_emg()
    print(f"min EMG={EMG.pEMG.min()}, max EMG={EMG.pEMG.max()}")
    emg_min, emg_max = emg_hist_lims[animal_name]
    ax[0].hist(EMG['pEMG'], bins=np.linspace(emg_min, emg_max + 1 / nbins, nbins), label=session_type)
    ax[0].set_xlabel("EMG Value")
    ax[0].set_ylabel("Count")
    sns.despine(ax=ax[ids])

match_axis_lims(ax, "x")
ax[0].legend()
plt.show()
#fig.savefig(animal_dir / f"{animal_name}_hypnograms.pdf")
