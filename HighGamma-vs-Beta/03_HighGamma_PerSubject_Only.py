# =========================================================
# HIGH-GAMMA BAND DYNAMICS (70–150 Hz) — Per-Subject Analysis
# Face trials only | Hit (Perceived) = Blue | Miss (Unperceived) = Red
# Stratified by visual noise: Low (0–30%) / Medium (35–65%) / High (70–100%)
# Analysis window highlighted: 200–600 ms post-stimulus
#
# Generates a 1x3 subplot grid per subject displaying baseline-normalized
# High-Gamma power envelope in face-selective ventral temporal cortex.
# High-Gamma serves as an index of local asynchronous neuronal population firing.
# =========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import gaussian_filter1d

# Path to Miller ECoG dataset file
DATA_PATH = 'faceshouses.npz'
if 'alldat' not in globals():
    alldat = np.load(DATA_PATH, allow_pickle=True)['dat']

# Experimental parameters and timing constants
FS = 1000                   # 1000 Hz acquisition rate
RESPONSE_WINDOW_MS = 1200   # Behavioral keypress reaction window [0, 1200) ms
EPOCH_PRE_MS = 300          # Pre-onset extraction window (ms)
EPOCH_POST_MS = 900         # Post-onset extraction window (ms)
PLOT_START_MS = -200        # Display time window start (ms)
PLOT_END_MS = 800           # Display time window end (ms)
BASELINE_START_MS = -200    # Pre-stimulus baseline interval start (ms)
BASELINE_END_MS = 0         # Pre-stimulus baseline interval end (ms)
SMOOTH_SIGMA = 8            # Gaussian smoothing sigma (ms)

# Visual noise stratification bins
NOISE_BINS = {
    'Low': (0, 30),
    'Medium': (35, 65),
    'High': (70, 100)
}

# Category-selective ventral temporal electrode assignments per subject
subjects_electrodes = {
    1: [0, 25, 30, 35, 39],
    4: [23, 24],
    5: [15],
    6: [6, 9, 15, 16, 17, 23, 24, 25, 30],
    7: [8, 9, 36, 38, 39]
}

# Directory to store generated figures
SAVE_DIR = 'HighGamma_PerSubject_Plots'
os.makedirs(SAVE_DIR, exist_ok=True)


def extract_high_gamma_power(V):
    """
    Extract High-Gamma (70-150 Hz) power envelope from continuous ECoG signal.
    Applies 4th-order Butterworth bandpass filter in SOS format followed by
    Hilbert transform to obtain the instantaneous analytic signal power.
    """
    sos = butter(
        4,
        [70, 150],
        btype='bandpass',
        fs=FS,
        output='sos'
    )

    filtered = sosfiltfilt(sos, V, axis=0)
    analytic = hilbert(filtered, axis=0)
    return np.abs(analytic) ** 2


def classify_trials(dat):
    """Classify trials as Hit (True) if keypress occurred within response window, else Miss (False)."""
    t_on = np.asarray(dat['t_on']).flatten().astype(int)
    key_press = np.asarray(dat['key_press']).flatten()

    hit = np.zeros(len(t_on), dtype=bool)
    for i, onset in enumerate(t_on):
        hit[i] = np.any(
            (key_press >= onset) &
            (key_press < onset + RESPONSE_WINDOW_MS)
        )
    return hit


def epoch_signal(data, t_on):
    """Segment continuous power signal into trial epochs [-300, +900] ms relative to stimulus onset."""
    pre = int(EPOCH_PRE_MS * FS / 1000)
    post = int(EPOCH_POST_MS * FS / 1000)
    time_vec = np.arange(-pre, post + 1) / FS

    epochs = []
    valid_idx = []

    for i, onset in enumerate(t_on):
        start = onset - pre
        stop = onset + post + 1
        if start >= 0 and stop <= data.shape[0]:
            epochs.append(data[start:stop, :])
            valid_idx.append(i)

    epochs = np.asarray(epochs)
    epochs = np.transpose(epochs, (0, 2, 1))
    return epochs, np.asarray(valid_idx), time_vec


def baseline_zscore(epochs, time_vec):
    """Z-score standardize trial epochs against pre-stimulus baseline window (-200 to 0 ms)."""
    bl = (
        (time_vec >= BASELINE_START_MS / 1000) &
        (time_vec < BASELINE_END_MS / 1000)
    )

    mu = np.mean(epochs[:, :, bl], axis=-1, keepdims=True)
    sd = np.std(epochs[:, :, bl], axis=-1, keepdims=True)
    sd = np.where(sd < 1e-8, np.nan, sd)

    return (epochs - mu) / sd


# Iterate through each subject and generate 1x3 High-Gamma dynamics figures
for subj_id, face_chans in subjects_electrodes.items():
    print(f'Processing Subject {subj_id}...')

    # dat corresponds to Experiment 2 (noisy perception task)
    dat = alldat[subj_id - 1][1]
    V = np.asarray(dat['V'], dtype=np.float32)
    t_on = np.asarray(dat['t_on']).flatten().astype(int)
    stim_cat = np.asarray(dat['stim_cat']).flatten()
    stim_noise = np.asarray(dat['stim_noise']).flatten()

    hit_all = classify_trials(dat)
    hg = extract_high_gamma_power(V)
    epochs, valid_idx, time_vec = epoch_signal(hg, t_on)
    epochs_z = baseline_zscore(epochs, time_vec)

    # Average across face-selective electrodes and apply Gaussian smoothing
    valid_chans = [c for c in face_chans if 0 <= c < V.shape[1]]
    roi = np.nanmean(epochs_z[:, valid_chans, :], axis=1)
    roi = gaussian_filter1d(roi, sigma=SMOOTH_SIGMA, axis=-1)

    face = stim_cat[valid_idx] == 2
    noise = stim_noise[valid_idx]
    hit = hit_all[valid_idx]

    # Create 1x3 subplot grid (one panel per noise bin)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)

    for ax, (noise_name, (lo, hi)) in zip(axes, NOISE_BINS.items()):
        noise_mask = (noise >= lo) & (noise <= hi)

        hit_mask = face & noise_mask & hit
        miss_mask = face & noise_mask & (~hit)

        # Plot Hit (blue) vs Miss (red) with SEM bands
        for mask, label, color in [
            (hit_mask, 'Hit', 'royalblue'),
            (miss_mask, 'Miss', 'crimson')
        ]:
            n = np.sum(mask)
            if n == 0:
                continue

            x = roi[mask]
            mean_trace = np.nanmean(x, axis=0)

            ax.plot(
                time_vec * 1000,
                mean_trace,
                color=color,
                lw=2.3,
                label=f'{label} (n={n})'
            )

            if n > 1:
                sem = np.nanstd(x, axis=0, ddof=1) / np.sqrt(n)
                ax.fill_between(
                    time_vec * 1000,
                    mean_trace - sem,
                    mean_trace + sem,
                    color=color,
                    alpha=0.20
                )

        ax.axvline(0, color='black', linestyle='--', alpha=0.6)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
        # Highlight 200-600 ms critical post-perceptual analysis window
        ax.axvspan(200, 600, color='gray', alpha=0.08)
        ax.set_xlim(PLOT_START_MS, PLOT_END_MS)
        ax.set_title(f'{noise_name} Noise ({lo}-{hi}%)', fontweight='bold')
        ax.set_xlabel('Time (ms)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('High-Gamma power (baseline z-score)')
    fig.suptitle(f'Subject {subj_id} — High-Gamma: Hit vs Miss', fontweight='bold')
    plt.tight_layout()

    filename = os.path.join(SAVE_DIR, f'Subject_{subj_id}_HighGamma_Hit_vs_Miss.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f'Saved: {filename}')
