from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
from neuropy.io.sleepscoremasterio import SleepScoreIO
from neuropy.plotting.epochs import plot_hypnogram
from neuropy.core.epoch import Epoch
from neuropy.utils.plot_util import match_axis_lims
import seaborn as sns

rec_dir = Path(r"D:\data\Nat\Psilocybin\Recording_Rats")
sessions = ["saline1", "psilocybin", "saline2"]
animal_name = "Rey"
animal_dir = rec_dir / animal_name
fig, ax = plt.subplots(1, 3, figsize=(11.3, 1.2))
fig.suptitle(animal_name)
ax[0].set_title("Saline1")
ax[1].set_title("Psilocybin")
ax[2].set_title("Saline2")


for ids, session_name in enumerate(sessions):
    base_dir = sorted(animal_dir.glob(f"*_{session_name}"))[0]

    sleep = SleepScoreIO(base_dir)
    brainstates = sleep.read_states(plot_states=False)

    try:
        inj_epochs = Epoch(epochs=None, file=sorted(base_dir.glob("*.injection.npy"))[0])
        inj_time = inj_epochs["POST"].starts[0]
    except IndexError:
        print(f"injection.npy not found in {base_dir}")
        inj_time = None

    plot_hypnogram(brainstates, ax=ax[ids], annotate=True)
    if inj_time is not None:
        ax[ids].axvline(inj_time, color='red', linestyle='--', linewidth=1)
    ax[ids].set_xticks([0, 3600, 7200])
    ax[ids].set_xlabel("Time (s)")
    ax[ids].axis("on")
    sns.despine(ax=ax[ids], left=True)
    ax[ids].set_yticks([])

match_axis_lims(ax, "x")
plt.show()
fig.savefig(animal_dir / f"{animal_name}_hypnograms.pdf")
