import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from neuropy.io.sleepscoremasterio import SleepScoreIO
from neuropy.core import Epoch

# Date dictionary - UPDATE THESE FOR EACH ANIMAL AS NEEDED
date_dict = {
    "Rey": {
        "saline1": "2022_06_01",
        "psilocybin": "2022_06_02",
        "saline2": "2022_06_03"
    },
    "Finn": {
        "saline1": "2022_02_15",
        "psilocybin": "2022_02_17",
        "saline2": "2022_02_18"
    },
    "Rose": {
        "saline1": "2022_08_09",
        "psilocybin": "2022_08_10",
        "saline2": "2022_08_11"
    },
    "Finn2": {
        "saline1": "2023_05_24",
        "psilocybin": "2023_05_25",
        "saline2": "2023_05_26"
    }
}

# List of animals and sessions
animals = ["Rey", "Finn", "Rose", "Finn2"]
sessions = ["saline1", "psilocybin", "saline2"]

# Collect data
data = []
for animal in animals:
    for session in sessions:
        date = date_dict[animal][session]
        base_path = rf"D:\data\Nat\Psilocybin\Recording_Rats\{animal}\{date}_{session}"

        print(f"Processing {animal} {session}: {base_path}")

        base_dir = Path(base_path)

        try:
            # Load sleep states
            sleep = SleepScoreIO(base_dir)
            brainstates = sleep.read_states(plot_states=False)
            print(f"Loaded brainstates: {len(brainstates)} epochs")

            # Load injection time
            try:
                inj_epochs = Epoch(epochs=None, file=sorted(base_dir.glob("*.injection.npy"))[0])
                inj_time = inj_epochs["POST"].starts[0]
                print(f"Injection time: {inj_time} seconds")
            except IndexError:
                print(f"injection.npy not found in {base_dir}")
                inj_time = None

            # Find latency to first NREM after injection
            if 'nrem' in brainstates.labels and inj_time is not None:
                nrem_starts = brainstates['nrem'].starts
                post_inj_starts = nrem_starts[nrem_starts >= inj_time]
                if len(post_inj_starts) > 0:
                    first_onset = post_inj_starts.min() - inj_time  # Latency in seconds
                    print(f"Latency to first NREM after injection: {first_onset} seconds")
                else:
                    first_onset = np.nan
                    print("No NREM after injection")
            else:
                first_onset = np.nan
                if 'nrem' not in brainstates.labels:
                    print("No NREM epochs found")
                if inj_time is None:
                    print("No injection time found")

            # Append to data
            data.append({
                "animal": animal,
                "session": session,
                "onset_time": first_onset
            })
        except Exception as e:
            print(f"Error processing {base_path}: {e}")
            continue

print(f"Collected data for {len(data)} entries")

# Create DataFrame
df = pd.DataFrame(data)
print(f"DataFrame shape: {df.shape}")
print(df.head())

# Plot using seaborn stripplot
if df.empty:
    print("No data collected to plot")
else:
    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df, x="session", y="onset_time", hue="animal", dodge=True)
    plt.title("Latency to First NREM Sleep Onset After Injection")
    plt.xlabel("Session")
    plt.ylabel("Latency (seconds)")
    plt.legend(title="Animal")
    plt.savefig(Path(r"D:\data\Nat\Psilocybin\Recording_Rats") / "SleepOnset_time.pdf")
    plt.show()
