from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
from neuropy.io.sleepscoremasterio import SleepScoreIO
from neuropy.plotting.epochs import plot_hypnogram
from neuropy.utils.plot_util import match_axis_lims
import seaborn as sns
alt_dir = Path(r"D:\data\Nat\Alternation\Recording_Rats")
session_name = "alternation"
animal_names = ["Finn", "Rey", "Rose"]
fig, ax = plt.subplots(1, 3, figsize=(11.3, 1.2))
fig.suptitle("Alternation")
ax[0].set_title("Finn")
ax[1].set_title("Rey")
ax[2].set_title("Rose")
for ids, animal_name in enumerate(animal_names):
    base_dir = sorted((alt_dir / animal_name).glob(f"*_{session_name}*"))[0]
    sleep = SleepScoreIO(base_dir)
    brainstates = sleep.read_states(plot_states=False)
    plot_hypnogram(brainstates, ax=ax[ids], annotate=True)
    ax[ids].set_xticks([0, 3600, 7200, 10800, 14400])
    ax[ids].set_xlabel("Time (s)")
    ax[ids].axis("on")
    sns.despine(ax=ax[ids], left=True)
    ax[ids].set_yticks([])
match_axis_lims(ax, "x")
plt.show()
fig.savefig(base_dir / f"{animal_name}_alternation_hypnograms.pdf")
