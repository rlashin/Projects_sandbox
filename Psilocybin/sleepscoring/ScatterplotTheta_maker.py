from pathlib import Path
import matplotlib.pyplot as plt
from Scatterplot_comp import SleepScoreMetricsIO
import seaborn as sns

# Define directories and animal
primary_dir = Path(r"D:\data\Nat\Psilocybin\Recording_Rats")
secondary_dir = Path(r"D:\data\Nat\Alternation\Recording_Rats")
animal_name = "Rey"

# Create figure with four subplots side by side
fig, ax = plt.subplots(1, 4, figsize=(16, 4), layout="tight")
sessions = ["alternation*", "psilocybin", "saline1", "saline2"]
titles = ["Alternation", "Psilocybin", "Saline1", "Saline2"]
base_dirs = [secondary_dir, primary_dir, primary_dir, primary_dir]  # alternation from alt_dir, others from psilo_dir

# Loop through sessions and plot scatterplots
for idx, (base_dir, session_type, title) in enumerate(zip(base_dirs, sessions, titles)):
    sess_dir = sorted((base_dir / animal_name).glob(f"*_{session_type}"))[0]

    # Load metrics data
    metrics_io = SleepScoreMetricsIO(sess_dir)
    metrics_df = metrics_io.read_metrics()

    # Plot scatterplot: EMG vs broadband slow wave
    sns.scatterplot(data=metrics_df, x='theta', y='EMG', ax=ax[idx], alpha=0.8, s=1)
    ax[idx].set_title(title)
    ax[idx].set_xlabel("Theta")
    ax[idx].set_ylabel("EMG")

    # Despine for cleaner look
    sns.despine(ax=ax[idx])

# Save the figure
fig.savefig(primary_dir / f"{animal_name}_scatterplot_Theta.pdf")
plt.show()
