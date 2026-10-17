# =========================================================================
# Event-Related Potential (ERP) Dynamics: Perceived vs. Unperceived Faces
# Under Parametric Visual Degradation (Miller ECoG Dataset)
#
# This script performs:
# 1. Subject data extraction & bad channel rejection (flat / extreme variance).
# 2. Common Average Referencing (CAR) and 0.5 Hz high-pass filtering for drift removal.
# 3. Behavioral classification: "Perceived" (Hit) vs. "Not Perceived" (Miss) based on keypress.
# 4. Baseline correction (-200 to 0 ms) and extraction of ventral temporal face-selective ERPs.
# 5. Grand-average ERP comparison across all face trials highlighting the N170/N200 window.
# 6. Noise-stratified analysis across 3 bins (Low: 0-30%, Medium: 35-65%, High: 70-100%).
# 7. Within-subject class balancing (subsampling) to verify robust perceptual differences.
# =========================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import sem

# =========================================================================
# Setup & Definitions
# =========================================================================
# Valid subjects with recorded behavioral keypress responses (5 out of 7 subjects)
valid_sub_ids = [0, 3, 4, 5, 6]

# Face-selective electrode channel indices per subject (localized in ventral temporal cortex)
face_channels_per_sub = {
    0: [0, 25, 30, 35, 39],
    3: [23, 24],
    4: [15],
    5: [6, 9, 15, 16, 17, 23, 24, 25, 30],
    6: [8, 9, 36, 38, 39],
}

# Sampling frequency in Hz
fs = 1000

# Extended epoch window in samples (pre-stimulus: 500 ms, post-stimulus: 1100 ms) to avoid filter edge artifacts
n_pre_ext, n_post_ext = 500, 1100
time_vec_ext = np.arange(-n_pre_ext, n_post_ext)

# Baseline window mask: -200 ms to 0 ms relative to stimulus onset
base_mask = (time_vec_ext >= -200) & (time_vec_ext < 0)

# Display/plotting window mask: -200 ms to +500 ms relative to stimulus onset
plot_mask = (time_vec_ext >= -200) & (time_vec_ext <= 500)
time_plot = time_vec_ext[plot_mask]

# 4th-order Butterworth high-pass filter at 0.5 Hz cutoff to remove slow DC drift
b_hp, a_hp = butter(4, 0.5 / (fs / 2), btype="high")

# Define condition colors
color_perceived = "#1f77b4"  # Blue
color_not_perceived = "#d62728"  # Red

records = []
trial_traces_list = []

# =========================================================================
# 1. Processing Loop & Perceptual Classification
# =========================================================================
# Iterate through each subject with valid behavioral data
for sub_id in valid_sub_ids:
    # dat2 corresponds to Experiment 2: Noisy Face Perception Task
    dat2 = alldat[sub_id][1]

    # Bad channel rejection based on variance criteria
    V_32 = dat2["V"].astype(np.float32)
    chan_stds = np.std(V_32, axis=0)
    # Flag flat channels with near-zero variance (< 1e-3)
    flat_chans = np.where(chan_stds < 1e-3)[0]
    # Flag excessively noisy channels (> 3.5 z-score of channel standard deviations)
    z_stds = np.abs((chan_stds - np.mean(chan_stds)) / (np.std(chan_stds) + 1e-8))
    noisy_chans = np.where(z_stds > 3.5)[0]
    bad_chans = np.unique(np.concatenate([flat_chans, noisy_chans]))
    good_chans = np.array([c for c in range(V_32.shape[1]) if c not in bad_chans])

    # Preprocessing: CAR (Common Average Referencing across good channels) + High-Pass Filter (0.5 Hz)
    V_car = V_32[:, good_chans] - np.mean(
        V_32[:, good_chans], axis=1, keepdims=True
    )
    V_filt = filtfilt(b_hp, a_hp, V_car, axis=0)

    # Map face-selective channels to the indices in the filtered good_chans array
    mapped_face_chans = [
        np.where(good_chans == ch)[0][0]
        for ch in face_channels_per_sub[sub_id]
        if ch in good_chans
    ]

    if not mapped_face_chans:
        continue  # Skip if all face channels were flagged as bad

    # Extract stimulus onset timestamps, category (1=house, 2=face), and visual noise level
    t_on = np.squeeze(dat2["t_on"]).astype(int)
    stim_cat = np.squeeze(dat2["stim_cat"])
    stim_noise = np.squeeze(dat2["stim_noise"])

    # Keypress lookup optimization
    kp_raw = np.squeeze(dat2["key_press"])
    is_continuous_kp = kp_raw.ndim == 1 and len(kp_raw) > 0 and kp_raw.max() <= 1

    for i in range(len(t_on)):
        # Target face trials (stim_cat == 2) within valid boundary
        if (
            stim_cat[i] == 2
            and (t_on[i] - n_pre_ext >= 0)
            and (t_on[i] + n_post_ext <= V_filt.shape[0])
        ):
            # Extract trial epoch and perform baseline correction (-200 to 0 ms)
            trial_epoch = V_filt[t_on[i] - n_pre_ext : t_on[i] + n_post_ext, :]
            baseline = np.mean(trial_epoch[base_mask, :], axis=0, keepdims=True)
            trial_clean = trial_epoch - baseline

            # Average signal across face-selective channels in ventral temporal cortex
            face_trace = np.mean(trial_clean[:, mapped_face_chans], axis=1)

            # Fast direct slice check for keypress in 0-1200 ms post-onset window
            if is_continuous_kp:
                kp_in_window = np.any(kp_raw[t_on[i] : t_on[i] + 1200] > 0)
            else:
                kp_in_window = np.any(
                    (kp_raw >= t_on[i]) & (kp_raw < t_on[i] + 1200)
                )

            # Record trial metadata and behavioral perception outcome
            records.append({
                "sub_id": f"Sub_{sub_id + 1}",
                "trial_idx": i,
                "noise_level": stim_noise[i],
                "perceived": "Perceived" if kp_in_window else "Not Perceived",
            })
            trial_traces_list.append(face_trace[plot_mask])

# Assemble dataframe of trial records and matrix of ERP traces
df_perception = pd.DataFrame(records)
trial_traces_matrix = np.array(trial_traces_list)

# =========================================================================
# 2. Grand Average ERP Plot (All Subjects Pooled)
# =========================================================================
# Create boolean masks for Perceived vs. Not Perceived trials across all collected trials
perc_mask = (df_perception["perceived"] == "Perceived").values
not_perc_mask = (df_perception["perceived"] == "Not Perceived").values

# Calculate mean voltage timecourses and Standard Error of the Mean (SEM) for both conditions
mean_perc = np.mean(trial_traces_matrix[perc_mask, :], axis=0)
sem_perc = sem(trial_traces_matrix[perc_mask, :], axis=0)

mean_not_perc = np.mean(trial_traces_matrix[not_perc_mask, :], axis=0)
sem_not_perc = sem(trial_traces_matrix[not_perc_mask, :], axis=0)

plt.figure(figsize=(9, 5.5), dpi=150)

# Shade the canonical N170/N200 physiological time window (200-280 ms post-stimulus)
plt.axvspan(
    200, 280, color="gray", alpha=0.15, label="N170 Window (200–280 ms)"
)
# Reference gridlines for time 0 (stimulus onset) and voltage 0 uV
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
plt.axvline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)

# Plot Perceived (Hit) ERP waveform with shaded ±1 SEM envelope
plt.plot(
    time_plot,
    mean_perc,
    color=color_perceived,
    linewidth=2.2,
    label=f"Perceived Face (n={np.sum(perc_mask)})",
)
plt.fill_between(
    time_plot,
    mean_perc - sem_perc,
    mean_perc + sem_perc,
    color=color_perceived,
    alpha=0.2,
)

# Plot Not Perceived (Miss) ERP waveform with shaded ±1 SEM envelope
plt.plot(
    time_plot,
    mean_not_perc,
    color=color_not_perceived,
    linewidth=2.2,
    label=f"Not Perceived Face (n={np.sum(not_perc_mask)})",
)
plt.fill_between(
    time_plot,
    mean_not_perc - sem_not_perc,
    mean_not_perc + sem_not_perc,
    color=color_not_perceived,
    alpha=0.2,
)

# Format figure axes, labels, and clean aesthetics
plt.xlabel("Time relative to stimulus onset (ms)", fontsize=11)
plt.ylabel("Voltage (µV)", fontsize=11)
plt.title(
    "Ventral Temporal ERP: Perceived vs. Unperceived Faces",
    fontsize=13,
    fontweight="bold",
)
plt.xlim(-200, 500)
plt.grid(True, linestyle="--", alpha=0.35)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.legend(loc="lower left", fontsize=9.5, frameon=True)
plt.tight_layout()
plt.show()

# -------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

# Define condition colors
color_perceived = "#1f77b4"      # Blue
color_not_perceived = "#d62728"  # Red

# =========================================================================
# 1. Apply 3-Bin Custom Noise Binning to df_perception
# =========================================================================
# Normalize noise values to 0-100% scale if stored as fractions [0, 1]
scale_factor = 100.0 if df_perception["noise_level"].max() <= 1.0 else 1.0
df_perception["noise_pct"] = df_perception["noise_level"] * scale_factor

# Categorize visual noise into 3 discrete bins: Low (0-30%), Medium (35-65%), High (70-100%)
bin_edges = [-0.1, 32.5, 67.5, 100.1]
bin_labels = ["0–30%", "35–65%", "70–100%"]
df_perception["noise_bin"] = pd.cut(
    df_perception["noise_pct"], bins=bin_edges, labels=bin_labels
)

# =========================================================================
# 2. Plot Grand-Average 1x3 Grid (All 5 Subjects Pooled)
# =========================================================================
# Initialize a 1x3 subplot grid to compare ERP dynamics across noise levels side by side
fig, axes = plt.subplots(
    1, 3, figsize=(15, 4.5), sharex=True, sharey=True, dpi=150
)

# Iterate across each noise bin (Low, Medium, High)
for b_idx, label in enumerate(bin_labels):
    ax = axes[b_idx]
    bin_mask = (df_perception["noise_bin"] == label).values

    # Isolate Perceived and Not Perceived trials within the current noise bin
    p_mask = bin_mask & (df_perception["perceived"] == "Perceived").values
    np_mask = bin_mask & (df_perception["perceived"] == "Not Perceived").values

    n_p = np.sum(p_mask)
    n_np = np.sum(np_mask)

    # Reference N170 shading (neutral gray) and zero reference axes
    ax.axvspan(200, 280, color="gray", alpha=0.15)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)

    # Grand-average Perceived face ERP trace + SEM band (Blue)
    if n_p > 0:
        mean_p = np.mean(trial_traces_matrix[p_mask, :], axis=0)
        sem_p = sem(trial_traces_matrix[p_mask, :], axis=0)
        ax.plot(
            time_plot,
            mean_p,
            color=color_perceived,
            linewidth=2.0,
            label=f"Perceived (n={n_p})",
        )
        ax.fill_between(
            time_plot,
            mean_p - sem_p,
            mean_p + sem_p,
            color=color_perceived,
            alpha=0.2,
        )

    # Grand-average Not Perceived face ERP trace + SEM band (Red)
    if n_np > 0:
        mean_np = np.mean(trial_traces_matrix[np_mask, :], axis=0)
        sem_np = sem(trial_traces_matrix[np_mask, :], axis=0)
        ax.plot(
            time_plot,
            mean_np,
            color=color_not_perceived,
            linewidth=2.0,
            label=f"Not Perceived (n={n_np})",
        )
        ax.fill_between(
            time_plot,
            mean_np - sem_np,
            mean_np + sem_np,
            color=color_not_perceived,
            alpha=0.2,
        )

    ax.set_title(
        f"Noise Bin: {label}", fontsize=11, fontweight="bold", loc="left"
    )
    ax.set_xlim(-200, 500)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True)

# Shared axis labels
for ax in axes:
    ax.set_xlabel("Time relative to onset (ms)", fontsize=10.5)
axes[0].set_ylabel("Voltage (µV)", fontsize=10.5)

plt.suptitle(
    "Grand Average Ventral Temporal ERP: Perceived vs. Unperceived Across Noise Bins (All 5 Subjects Pooled)",
    y=1.05,
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

# -------------------------------

import pandas as pd

# =========================================================================
# 1. Ensure 3-Bin Custom Noise Binning is Applied to df_perception
# =========================================================================
# Validate noise percentage scaling
scale_factor = 100.0 if df_perception["noise_level"].max() <= 1.0 else 1.0
df_perception["noise_pct"] = df_perception["noise_level"] * scale_factor

# Categorize visual noise into 3 bins
bin_edges = [-0.1, 32.5, 67.5, 100.1]
bin_labels = ["0–30%", "35–65%", "70–100%"]
df_perception["noise_bin"] = pd.cut(
    df_perception["noise_pct"], bins=bin_edges, labels=bin_labels
)

# =========================================================================
# 2. Cross-tabulate trial counts per subject, noise bin, and perception status
# =========================================================================
# Inspect trial counts to assess class imbalance (e.g. fewer Misses in Low noise, fewer Hits in High noise)
imbalance_table = pd.crosstab(
    [df_perception["sub_id"], df_perception["noise_bin"]],
    df_perception["perceived"],
    margins=True,
)

print("--- Trial Counts per Subject and 3 Noise Bins ---")
print(imbalance_table)

# -------------------------------

import numpy as np
import pandas as pd

# =========================================================================
# 1. Ensure 3-Bin Custom Noise Binning is Applied to df_perception
# =========================================================================
scale_factor = 100.0 if df_perception["noise_level"].max() <= 1.0 else 1.0
df_perception["noise_pct"] = df_perception["noise_level"] * scale_factor

bin_edges = [-0.1, 32.5, 67.5, 100.1]
bin_labels = ["0–30%", "35–65%", "70–100%"]
df_perception["noise_bin"] = pd.cut(
    df_perception["noise_pct"], bins=bin_edges, labels=bin_labels
)

# =========================================================================
# 2. Within-Subject & Within-Bin Random Subsampling (Class Balancing)
# =========================================================================
np.random.seed(42)  # Set seed for reproducible random subsampling

balanced_indices = []

# Group by subject and noise bin to balance within-subject conditions
for (sub_id, bin_label), group in df_perception.groupby(
    ["sub_id", "noise_bin"], observed=False
):
    perc_idx = group[group["perceived"] == "Perceived"].index
    not_perc_idx = group[group["perceived"] == "Not Perceived"].index

    # Find minority class count
    n_min = min(len(perc_idx), len(not_perc_idx))

    # Sample n_min trials from both conditions without replacement
    if n_min > 0:
        sampled_perc = np.random.choice(perc_idx, size=n_min, replace=False)
        sampled_not_perc = np.random.choice(
            not_perc_idx, size=n_min, replace=False
        )
        balanced_indices.extend(sampled_perc)
        balanced_indices.extend(sampled_not_perc)

# Extract balanced DataFrame and matrix subset
df_balanced = df_perception.loc[balanced_indices].copy()
balanced_traces_matrix = trial_traces_matrix[balanced_indices, :]

print(f"Original total trials: {len(df_perception)}")
print(f"Balanced total trials: {len(df_balanced)}")

# Verify balanced counts
print("\n--- Balanced Trial Counts across 3 Noise Bins ---")
print(
    pd.crosstab(
        [df_balanced["sub_id"], df_balanced["noise_bin"]],
        df_balanced["perceived"],
    )
)

# -------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

# Define condition colors
color_perceived = "#1f77b4"      # Blue
color_not_perceived = "#d62728"  # Red

# =========================================================================
# 1. Ensure 3-Bin Custom Noise Binning
# =========================================================================
bin_labels = ["0–30%", "35–65%", "70–100%"]

# =========================================================================
# 2. Plot Balanced Grand-Average 1x3 Grid
# =========================================================================
# Plot grand average ERPs using class-balanced trials to confirm amplitude effects
fig, axes = plt.subplots(
    1, 3, figsize=(15, 4.5), sharex=True, sharey=True, dpi=150
)

for b_idx, label in enumerate(bin_labels):
    ax = axes[b_idx]
    bin_mask = (df_balanced["noise_bin"] == label).values

    p_mask = bin_mask & (df_balanced["perceived"] == "Perceived").values
    np_mask = bin_mask & (df_balanced["perceived"] == "Not Perceived").values

    n_p = np.sum(p_mask)
    n_np = np.sum(np_mask)

    # Neutral N170 shading and zero reference axes
    ax.axvspan(200, 280, color="gray", alpha=0.15)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)

    # Plot balanced Perceived traces (Blue)
    if n_p > 0:
        mean_p = np.mean(balanced_traces_matrix[p_mask, :], axis=0)
        sem_p = sem(balanced_traces_matrix[p_mask, :], axis=0)
        ax.plot(
            time_plot,
            mean_p,
            color=color_perceived,
            linewidth=2.0,
            label=f"Perceived (n={n_p})",
        )
        ax.fill_between(
            time_plot,
            mean_p - sem_p,
            mean_p + sem_p,
            color=color_perceived,
            alpha=0.2,
        )

    # Plot balanced Not Perceived traces (Red)
    if n_np > 0:
        mean_np = np.mean(balanced_traces_matrix[np_mask, :], axis=0)
        sem_np = sem(balanced_traces_matrix[np_mask, :], axis=0)
        ax.plot(
            time_plot,
            mean_np,
            color=color_not_perceived,
            linewidth=2.0,
            label=f"Not Perceived (n={n_np})",
        )
        ax.fill_between(
            time_plot,
            mean_np - sem_np,
            mean_np + sem_np,
            color=color_not_perceived,
            alpha=0.2,
        )

    ax.set_title(
        f"Noise Bin: {label}", fontsize=11, fontweight="bold", loc="left"
    )
    ax.set_xlim(-200, 500)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True)

# Shared axis labels
for ax in axes:
    ax.set_xlabel("Time relative to onset (ms)", fontsize=10.5)
axes[0].set_ylabel("Voltage (µV)", fontsize=10.5)

plt.suptitle(
    "Balanced Grand Average ERP: Perceived vs. Unperceived Across 3 Noise Bins",
    y=1.05,
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.show()