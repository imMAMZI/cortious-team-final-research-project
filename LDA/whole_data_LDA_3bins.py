# =========================================================================
# Linear Discriminant Analysis (LDA) Decoding on Raw Voltage (Whole Data)
# Pillar II: Spontaneous Neural Decoding of Perceptual Content
#
# This script decodes stimulus category (Face vs. House) from raw ECoG voltage
# across ALL recorded electrodes (without frequency filtering) across 3 noise bins:
#   - Low Noise    (0–30%)
#   - Medium Noise (35–65%)
#   - High Noise   (70–100%)
#
# Two decoding approaches are implemented:
# 1. Static Window Decoding: Mean voltage in the N170 window (200–280 ms)
#    using 5-fold Stratified Cross-Validation with shrinkage LDA.
# 2. Time-Resolved Decoding: Sliding 20 ms window (-200 to +500 ms) tracking
#    the dynamic emergence and collapse of decoding accuracy over time.
# =========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load dataset containing all subjects and experimental sessions
print("Loading dataset...")
alldat = np.load('faceshouses.npz', allow_pickle=True)['dat']

# Subjects with valid behavioral keypress recordings in Experiment 2
valid_sub_ids = [0, 3, 4, 5, 6]

# Epoching parameters (in milliseconds relative to stimulus onset)
n_pre = 200   # Pre-stimulus baseline duration (ms)
n_post = 500  # Post-stimulus duration (ms)
base_mask = slice(0, 200)                   # Baseline interval: -200 ms to 0 ms (indices 0 to 200)
erp_mask = slice(200 + 200, 200 + 280)     # Static N170 ERP window: 200 to 280 ms post-onset
time_vec = np.arange(-n_pre, n_post)        # Time vector spanning -200 to +499 ms

def get_3_bins(noise):
    """
    Discretize continuous visual noise percentages into 3 standardized bins:
      Bin 1: Low Noise    (0–30%)
      Bin 2: Medium Noise (35–65%)
      Bin 3: High Noise   (70–100%)
    """
    # Some instances of noise could be formatted as strings '30%', extracting digit
    noise = float(''.join(filter(str.isdigit, str(noise))) or 0)
    if noise <= 30:
        return 1
    elif noise <= 65:
        return 2
    else:
        return 3

# =========================================================================
# 1. Static ERP-Window Decoding (All Channels, No Freq Filters, 3 Bins)
# =========================================================================
print("Starting Static LDA on WHOLE DATA (No frequency filtering, 3 custom bins)...")
decoding_results = []
subject_data = {} 

for sub_id in valid_sub_ids:
    # Experiment 2: Noisy face perception task
    dat2 = alldat[sub_id][1]
    
    # Common Average Referencing (CAR) across all electrodes
    V = dat2["V"].astype(np.float32)
    # CAR
    V_car = V - np.mean(V, axis=1, keepdims=True)
    
    # Extract trial onsets, stimulus categories (1=house, 2=face), and noise values
    t_on = np.squeeze(dat2["t_on"]).astype(int)
    stim_cat = np.squeeze(dat2["stim_cat"])
    stim_noise = np.squeeze(dat2["stim_noise"])
    
    # Select all valid house (1) and face (2) trials
    valid_trials = np.where((stim_cat == 1) | (stim_cat == 2))[0]
    
    epochs_list, y_sub, noise_sub = [], [], []
    for i in valid_trials:
        # Check boundary conditions to ensure full epoch fits within recording
        if (t_on[i] - n_pre >= 0) and (t_on[i] + n_post <= V_car.shape[0]):
            trial_epoch = V_car[t_on[i] - n_pre : t_on[i] + n_post, :]
            # Baseline correction (-200 to 0 ms)
            baseline = np.mean(trial_epoch[base_mask, :], axis=0, keepdims=True)
            trial_clean = trial_epoch - baseline
            
            epochs_list.append(trial_clean)
            # Binary label: 1 = Face, 0 = House
            y_sub.append(1 if stim_cat[i] == 2 else 0)
            noise_sub.append(get_3_bins(stim_noise[i]))
            
    epochs = np.array(epochs_list)
    y_sub = np.array(y_sub)
    noise_bins = np.array(noise_sub)
    
    # Cache epoched data for both static and subsequent time-resolved decoding
    subject_data[sub_id] = {
        "epochs": epochs,
        "y": y_sub,
        "noise_bins": noise_bins
    }
    
    # Feature Extraction: Mean voltage across the N170 window (200-280 ms) for each channel
    X_all = np.mean(epochs[:, erp_mask, :], axis=1)
    
    # Perform cross-validated LDA classification separately within each noise bin
    for bin_idx in range(1, 4):
        bin_mask = (noise_bins == bin_idx)
        X_bin = X_all[bin_mask]
        y_bin = y_sub[bin_mask]
        
        # Need at least two classes to do LDA cross_val
        if len(np.unique(y_bin)) < 2:
            continue
            
        # 5-fold Stratified Cross-Validation with regularized LDA (shrinkage)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        scores = cross_val_score(lda, X_bin, y_bin, cv=cv, scoring="accuracy")
        
        decoding_results.append({
            "sub_id": f"Sub_{sub_id + 1}",
            "noise_bin": bin_idx,
            "accuracy": np.mean(scores)
        })

# Aggregate decoding results into DataFrame
df_lda = pd.DataFrame(decoding_results)

# Plotting Static LDA Results: Individual Subject Curves + Grand Average
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
stats = df_lda.groupby("noise_bin")["accuracy"].agg(["mean", "sem"]) * 100
bins = stats.index.to_numpy()

unique_subs = df_lda["sub_id"].unique()
colors = plt.cm.tab10(np.linspace(0, 0.8, len(unique_subs)))

# Plot individual subject decoding trajectories
for sub_id, color in zip(unique_subs, colors):
    sub_df = df_lda[df_lda["sub_id"] == sub_id]
    ax.plot(sub_df["noise_bin"], sub_df["accuracy"] * 100, alpha=0.45, linestyle="--", marker="o", markersize=4, color=color, label=sub_id)

# Plot Grand Average trajectory with ±1 SEM error band
ax.plot(bins, stats["mean"], color="crimson", linewidth=2.8, marker="s", markersize=6, label="Grand Average")
ax.fill_between(bins, stats["mean"] - stats["sem"], stats["mean"] + stats["sem"], color="crimson", alpha=0.18, label="±1 SEM")
# Theoretical chance level is 50% for balanced two-class classification
ax.axhline(50, color="black", linestyle=":", linewidth=1.2, label="Chance Level (50%)")
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Low (0-30)", "Medium (35-65)", "High (70-100)"])
ax.set_xlabel("Visual Noise Level")
ax.set_ylabel("Face vs. House Decoding Accuracy (%)")
ax.set_title("Static LDA (ALL Electrodes, 3 Custom Bins)", fontweight="bold")
ax.set_ylim(35, 105)
ax.legend(loc="upper right", fontsize=8.5, ncol=2)
plt.tight_layout()
plt.savefig("whole_data_3bins_static_lda.png")
print("Saved whole_data_3bins_static_lda.png")

# =========================================================================
# 2. Time-Resolved LDA Decoding (Sliding Window, All Channels, 3 Bins)
# =========================================================================
print("Starting Time-Resolved LDA (3 custom bins)...")
# Sliding window parameters: 20 ms width, 10 ms step size (-200 to +500 ms)
time_steps = np.arange(-200, 505, 10)
window_width = 20
time_decoding_results = []

# Evaluate decoding timecourse across each noise bin independently
for bin_idx in range(1, 4):
    acc_over_time = np.zeros(len(time_steps))

    for t_idx, t_center in enumerate(time_steps):
        # Time window mask centered at t_center
        t_mask = (time_vec >= t_center - window_width // 2) & (time_vec <= t_center + window_width // 2)
        step_accs = []
        
        for sub_id in valid_sub_ids:
            sub = subject_data[sub_id]
            bin_mask = sub["noise_bins"] == bin_idx

            if np.sum(bin_mask) == 0:
                continue

            # Feature: mean voltage within the current 20 ms window across all electrodes
            X_step = np.mean(sub["epochs"][bin_mask][:, t_mask, :], axis=1)
            y_step = sub["y"][bin_mask]

            if len(np.unique(y_step)) < 2:
                continue

            # Ensure K_Folds works even if some subjects have very few trials in a bin
            min_class_counts = np.min(np.bincount(y_step))
            if min_class_counts < 2:
                continue
            k_folds = min(3, min_class_counts)
            
            # Cross-validated LDA on current time step
            cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            scores = cross_val_score(lda, X_step, y_step, cv=cv, scoring="accuracy")
            step_accs.append(np.mean(scores))

        # Average accuracy across subjects at current time step
        acc_over_time[t_idx] = np.mean(step_accs) if len(step_accs) > 0 else 0.5

    time_decoding_results.append((bin_idx, acc_over_time))

# Plot Time-Resolved Decoding Curves across the 3 Noise Bins
plt.figure(figsize=(10, 5), dpi=150)
colors = ['#1f77b4', '#ff7f0e', '#d62728'] # Blue, Orange, Red
bin_labels = {1: "Low (0-30)", 2: "Medium (35-65)", 3: "High (70-100)"}

for bin_idx, acc_curve in time_decoding_results:
    plt.plot(time_steps, acc_curve * 100, color=colors[bin_idx - 1], linewidth=2, label=bin_labels[bin_idx])

# Reference markers: stimulus onset (t=0 ms), 50% chance line, and N170 window
plt.axvline(0, color="black", linestyle="-", alpha=0.7)
plt.axhline(50, color="black", linestyle=":", label="Chance (50%)")
plt.axvspan(200, 280, color="red", alpha=0.12, label="N170 Window (200-280 ms)")

plt.xlabel("Time relative to stimulus onset (ms)", fontsize=11)
plt.ylabel("Decoding Accuracy (%)", fontsize=11)
plt.title("Time-Resolved Decoding (ALL Electrodes, 3 Custom Bins)", fontsize=13, fontweight="bold")
plt.legend(loc="upper right", fontsize=8.5, frameon=True)
plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig("whole_data_3bins_time_resolved_lda.png")
print("Saved whole_data_3bins_time_resolved_lda.png")
print("Done!")
