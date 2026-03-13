from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
from Scatterplot_comp import SleepScoreMetricsIO
import seaborn as sns
import pandas as pd
# Define directories and animal
primary_dir = Path(r"D:\data\Nat\Psilocybin\Recording_Rats")
secondary_dir = Path(r"D:\data\Nat\Alternation\Recording_Rats")
animal_name = "Finn2"

# Create figure with four subplots side by side
fig, ax = plt.subplots(1, 4, figsize=(16, 4), layout="tight")
sessions = ["alternation*", "saline1", "psilocybin", "saline2"]
titles = ["Alternation", "Saline1", "Psilocybin", "Saline2"]
base_dirs = [secondary_dir, primary_dir, primary_dir, primary_dir]

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
    metrics_io = SleepScoreMetricsIO(sess_dir)
    metrics_df = metrics_io.read_metrics()

    # Plot scatterplot: EMG vs broadband slow wave
    sns.scatterplot(data=metrics_df, x='slowwave', y='EMG', ax=ax[idx], alpha=0.8, s=1)
    ax[idx].set_title(title)
    ax[idx].set_xlabel("Broadband Slow Wave")
    ax[idx].set_ylabel("EMG")

    # Set standardized axis limits
    ax[idx].set_xlim(x_min, x_max)
    ax[idx].set_ylim(y_min, y_max)

    # Despine for cleaner look
    sns.despine(ax=ax[idx])

# Save the figure
fig.savefig(primary_dir / f"{animal_name}_scatterplot_SWS.pdf")
plt.show()
