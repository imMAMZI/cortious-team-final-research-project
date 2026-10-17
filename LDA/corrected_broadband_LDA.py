# =========================================================================
# Linear Discriminant Analysis (LDA) Decoding on True Broadband Power (70–150 Hz)
# Pillar II: Spontaneous Neural Decoding of Perceptual Content
#
# This script decodes stimulus category (Face vs. House) using Event-Related
# Broadband (ERBB / High-Gamma power: 70–150 Hz) across ALL recorded electrodes.
# Unlike raw ERP deflections which reflect synchronized feed-forward activity,
# broadband spectral power tracks asynchronous local neuronal firing and recurrent
# processing.
#
# Analysis consists of:
# 1. 70–150 Hz Bandpass filtering followed by Hilbert analytic signal power extraction.
# 2. Static Window Decoding: Mean broadband power in 50–400 ms post-stimulus window.
# 3. Time-Resolved Decoding: Sliding 20 ms window (-200 to +500 ms) tracking sustained
#    perceptual decoding across 3 noise bins: Low (0–30%), Medium (35–65%), High (70–100%).
# =========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load Miller ECoG dataset
print("Loading dataset...")
alldat = np.load('faceshouses.npz', allow_pickle=True)['dat']
valid_sub_ids = [0, 3, 4, 5, 6]  # Subjects with valid behavioral keypress data
fs = 1000                        # Sampling rate (1000 Hz)

# Epoching definitions (in milliseconds)
n_pre = 200   # Pre-stimulus baseline duration (ms)
n_post = 500  # Post-stimulus duration (ms)
base_mask = slice(0, 200)  # Baseline window: -200 to 0 ms relative to onset
# Broadband power typically starts earlier and lasts longer than the ERP. 
# We'll use 50ms to 400ms for the static window, which is standard for Miller broadband.
bb_mask = slice(200 + 50, 200 + 400) 
time_vec = np.arange(-n_pre, n_post)

def get_3_bins(noise):
    """Discretize continuous visual noise into Low (1), Medium (2), and High (3) bins."""
    noise = float(''.join(filter(str.isdigit, str(noise))) or 0)
    if noise <= 30: return 1
    elif noise <= 65: return 2
    else: return 3

# =========================================================================
# 1. Static ERP-Window Decoding (All Channels, TRUE Broadband)
# =========================================================================
print("Starting Static LDA on TRUE BROADBAND (70-150Hz power), all electrodes, 3 custom bins...")
decoding_results = []
subject_data = {} 

# Filter design for the broadband proxy (70-150 Hz)
b_bb, a_bb = butter(4, [70.0 / (fs / 2), 150.0 / (fs / 2)], btype='bandpass')

for sub_id in valid_sub_ids:
    # dat2: Experiment 2 (Phase-scrambled Face/House detection task)
    dat2 = alldat[sub_id][1]
    
    # Common Average Referencing (CAR) across all electrodes
    V = dat2["V"].astype(np.float32)
    V_car = V - np.mean(V, axis=1, keepdims=True)
    
    # Extract TRUE Broadband power using Hilbert envelope of the 70-150Hz band
    # This matches the Miller ECoG definition of Event-Related Broadband (ERBB)
    V_bb_filt = filtfilt(b_bb, a_bb, V_car, axis=0)
    V_bb_power = np.abs(hilbert(V_bb_filt, axis=0))**2
    
    t_on = np.squeeze(dat2["t_on"]).astype(int)
    stim_cat = np.squeeze(dat2["stim_cat"])
    stim_noise = np.squeeze(dat2["stim_noise"])
    
    # Filter for valid trials of houses (1) and faces (2)
    valid_trials = np.where((stim_cat == 1) | (stim_cat == 2))[0]
    
    epochs_list, y_sub, noise_sub = [], [], []
    for i in valid_trials:
        if (t_on[i] - n_pre >= 0) and (t_on[i] + n_post <= V_bb_power.shape[0]):
            # Epoch the POWER, not the raw voltage
            trial_epoch = V_bb_power[t_on[i] - n_pre : t_on[i] + n_post, :]
            # Baseline correct the power
            baseline = np.mean(trial_epoch[base_mask, :], axis=0, keepdims=True)
            trial_clean = trial_epoch - baseline
            
            epochs_list.append(trial_clean)
            # Binary classification label: 1 = Face, 0 = House
            y_sub.append(1 if stim_cat[i] == 2 else 0)
            noise_sub.append(get_3_bins(stim_noise[i]))
            
    epochs = np.array(epochs_list)
    y_sub = np.array(y_sub)
    noise_bins = np.array(noise_sub)
    
    # Store epoched broadband power for reuse in time-resolved analysis
    subject_data[sub_id] = {
        "epochs": epochs,
        "y": y_sub,
        "noise_bins": noise_bins
    }
    
    # Feature Extraction: Mean BROADBAND POWER in 50-400 ms across ALL channels
    X_all = np.mean(epochs[:, bb_mask, :], axis=1)
    
    # Cross-validated LDA separately for each of the 3 noise bins
    for bin_idx in range(1, 4):
        bin_mask = (noise_bins == bin_idx)
        X_bin = X_all[bin_mask]
        y_bin = y_sub[bin_mask]
        
        # Verify both classes exist in this bin
        if len(np.unique(y_bin)) < 2:
            continue
            
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        scores = cross_val_score(lda, X_bin, y_bin, cv=cv, scoring="accuracy")
        
        decoding_results.append({
            "sub_id": f"Sub_{sub_id + 1}",
            "noise_bin": bin_idx,
            "accuracy": np.mean(scores)
        })

df_lda = pd.DataFrame(decoding_results)

# Plotting Static Broadband LDA Results
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
stats = df_lda.groupby("noise_bin")["accuracy"].agg(["mean", "sem"]) * 100
bins = stats.index.to_numpy()

unique_subs = df_lda["sub_id"].unique()
colors = plt.cm.tab10(np.linspace(0, 0.8, len(unique_subs)))

# Plot individual subject curves
for sub_id, color in zip(unique_subs, colors):
    sub_df = df_lda[df_lda["sub_id"] == sub_id]
    ax.plot(sub_df["noise_bin"], sub_df["accuracy"] * 100, alpha=0.45, linestyle="--", marker="o", markersize=4, color=color, label=sub_id)

# Grand Average curve with ±1 SEM shading
ax.plot(bins, stats["mean"], color="crimson", linewidth=2.8, marker="s", markersize=6, label="Grand Average")
ax.fill_between(bins, stats["mean"] - stats["sem"], stats["mean"] + stats["sem"], color="crimson", alpha=0.18, label="±1 SEM")
# Chance line at 50%
ax.axhline(50, color="black", linestyle=":", linewidth=1.2, label="Chance Level (50%)")
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Low (0-30)", "Medium (35-65)", "High (70-100)"])
ax.set_xlabel("Visual Noise Level")
ax.set_ylabel("Face vs. House Decoding Accuracy (%)")
ax.set_title("Static LDA (True Broadband Power, All Electrodes)", fontweight="bold")
ax.set_ylim(35, 105)
ax.legend(loc="upper right", fontsize=8.5, ncol=2)
plt.tight_layout()
plt.savefig("corrected_broadband_static_lda.png")
print("Saved corrected_broadband_static_lda.png")


# =========================================================================
# 2. Time-Resolved LDA Decoding (Sliding Window, All Channels, TRUE Broadband)
# =========================================================================
print("Starting Time-Resolved LDA (True Broadband, 3 custom bins)...")
# Time-resolved sliding window: 20 ms width, 10 ms step (-200 to +500 ms)
time_steps = np.arange(-200, 505, 10)
window_width = 20
time_decoding_results = []

for bin_idx in range(1, 4):
    acc_over_time = np.zeros(len(time_steps))

    for t_idx, t_center in enumerate(time_steps):
        # Current sliding window mask
        t_mask = (time_vec >= t_center - window_width // 2) & (time_vec <= t_center + window_width // 2)
        step_accs = []
        
        for sub_id in valid_sub_ids:
            sub = subject_data[sub_id]
            bin_mask = sub["noise_bins"] == bin_idx

            if np.sum(bin_mask) == 0:
                continue

            # Feature: Mean broadband power within the sliding window
            X_step = np.mean(sub["epochs"][bin_mask][:, t_mask, :], axis=1)
            y_step = sub["y"][bin_mask]

            if len(np.unique(y_step)) < 2:
                continue

            # Ensure minimum class count for cross-validation splits
            min_class_counts = np.min(np.bincount(y_step))
            if min_class_counts < 2:
                continue
            k_folds = min(3, min_class_counts)
            
            # Stratified CV LDA on current time slice
            cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            scores = cross_val_score(lda, X_step, y_step, cv=cv, scoring="accuracy")
            step_accs.append(np.mean(scores))

        # Average across subjects at this time step
        acc_over_time[t_idx] = np.mean(step_accs) if len(step_accs) > 0 else 0.5

    time_decoding_results.append((bin_idx, acc_over_time))

# Plot Time-Resolved Broadband Decoding Timecourses
plt.figure(figsize=(10, 5), dpi=150)
colors = ['#1f77b4', '#ff7f0e', '#d62728'] 
bin_labels = {1: "Low (0-30)", 2: "Medium (35-65)", 3: "High (70-100)"}

for bin_idx, acc_curve in time_decoding_results:
    plt.plot(time_steps, acc_curve * 100, color=colors[bin_idx - 1], linewidth=2, label=bin_labels[bin_idx])

# Reference markers: stimulus onset, 50% chance, and 50-400 ms Broadband window
plt.axvline(0, color="black", linestyle="-", alpha=0.7)
plt.axhline(50, color="black", linestyle=":", label="Chance (50%)")
# Note: Broadband power shift typically begins earlier than the N170 ERP deflection.
plt.axvspan(50, 400, color="red", alpha=0.12, label="Broadband Window (50-400 ms)")

plt.xlabel("Time relative to stimulus onset (ms)", fontsize=11)
plt.ylabel("Decoding Accuracy (%)", fontsize=11)
plt.title("Time-Resolved Decoding (True Broadband Power, All Electrodes)", fontsize=13, fontweight="bold")
plt.legend(loc="upper right", fontsize=8.5, frameon=True)
plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig("corrected_broadband_time_resolved_lda.png")
print("Saved corrected_broadband_time_resolved_lda.png")
print("Done!")
