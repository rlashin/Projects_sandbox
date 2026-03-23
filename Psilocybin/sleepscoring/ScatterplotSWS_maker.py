from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
from Scatterplot_comp import SleepScoreMetricsIO
from neuropy.io.sleepscoremasterio import SleepScoreIO
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
# Define directories and animal
primary_dir = Path(r"D:\data\Nat\Psilocybin\Recording_Rats")
secondary_dir = Path(r"D:\data\Nat\Alternation\Recording_Rats")
animal_name = "Finn"

# Create figure with four subplots side by side
fig, ax = plt.subplots(1, 4, figsize=(10, 2.5), layout="tight")
sessions = ["alternation*", "saline1", "psilocybin", "saline2"]
titles = ["Alternation", "Saline1", "Psilocybin", "Saline2"]
base_dirs = [secondary_dir, primary_dir, primary_dir, primary_dir]

thresh_dict = {"Finn": {"sw": 1.0212, "theta": 0.5943, "emg": 0.0796},
               "Rose": {"sw": 1.0549, "theta": 0.48, "emg": 0.0466},
               "Rey": {"sw": 1.0493, "theta": 0.5784, "emg": 0.1068},
               "Finn2": {"sw": 0.7449, "theta": 0.4994, "emg": 0.1254}}

# Collect all data to compute global limits
sess_dirs = [sorted((base_dir / animal_name).glob(f"*_{session_type}"))[0] for base_dir, session_type in zip(base_dirs, sessions)]
all_df = pd.concat([SleepScoreMetricsIO(sess_dir).read_metrics() for sess_dir in sess_dirs])
x_min = all_df['slowwave'].min()
x_max = all_df['slowwave'].max()
y_min = all_df['EMG'].min()
y_max = all_df['EMG'].max()

# Loop through sessions and plot scatterplots
for idx, (base_dir, session_type, title) in enumerate(zip(base_dirs, sessions, titles)):
    sess_dir = sorted((base_dir / animal_name).glob(f"*_{session_type}"))[0]

    # Load metrics data
    # metrics_io = SleepScoreMetricsIO(sess_dir)
    # metrics_df = metrics_io.read_metrics()

    # Load sleep states (and metrics data)
    states_io = SleepScoreIO(sess_dir)
    states_epochs = states_io.read_states(plot_states=False)
    metrics_df = states_io.read_metrics()

    # Add sleepstate column to metrics_df
    inside_bool, _, sleepstates = states_epochs.contains(metrics_df['timestamps'].values)
    metrics_df['sleepstate'] = 'unknown'
    metrics_df.loc[inside_bool, 'sleepstate'] = sleepstates
    #Create Palette
    palette = {
    'active': 'blue',
    'rem': 'red',
    'nrem': 'green',
    'quiet': 'yellow',
    'QWakestate': 'yellow',
    'unknown': 'grey',
    }

    # Plot scatterplot: EMG vs broadband slow wave
    sns.scatterplot(data=metrics_df, x='slowwave', y='EMG', ax=ax[idx], rasterized=True, alpha=0.8, s=1, hue='sleepstate', palette=palette, legend=False)
    ax[idx].set_title(title)
    ax[idx].set_xlabel("Broadband Slow Wave")
    ax[idx].set_ylabel("EMG")

    # Set standardized axis limits
    ax[idx].set_xlim(x_min, x_max)
    ax[idx].set_ylim(y_min, y_max)

    # Despine for cleaner look
    sns.despine(ax=ax[idx])

   # Add threshold lines
    ax[idx].axvline(x=thresh_dict[animal_name]['sw'], color='black', linestyle='--')
    ax[idx].axhline(y=thresh_dict[animal_name]['emg'], color='black', linestyle='--')

    # Add custom legend
    legend_states = ['active', 'rem', 'nrem', 'quiet', 'unknown']
    patches = [mpatches.Patch(color=palette[state], label=state) for state in legend_states]
    fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=len(legend_states))

# Save the figure
fig.savefig(primary_dir / f"{animal_name}_scatterplot_SWS.pdf")
plt.show()
