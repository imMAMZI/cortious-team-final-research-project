# ==========================================================
# BETA GROUP AVERAGE ACROSS 5 SUBJECTS (13–30 Hz)
# Equal subject weighting (Within-subject averaging first)
# Hit (Perceived) = Blue | Miss (Unperceived) = Red
# No individual-subject lines
# Same 200–600 ms highlighted window as High-Gamma
#
# This script computes the grand-average Beta power dynamics across 5 subjects
# in face-selective ventral temporal cortex under Low, Medium, and High noise.
# By calculating condition means within-subject first before group averaging,
# each subject contributes equally to avoid trial-count bias.
# ==========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d

# Path to Miller ECoG dataset file
DATA_PATH = 'faceshouses.npz'
if 'alldat' not in globals():
    alldat = np.load(DATA_PATH, allow_pickle=True)['dat']

# Experimental constants and parameters
FS = 1000                           # 1000 Hz acquisition rate
VALID_SUBJECTS = [1, 4, 5, 6, 7]   # 1-indexed subjects with behavioral keypress data
RESPONSE_WINDOW_MS = 1200           # Behavioral response window [0, 1200) ms

EPOCH_PRE_MS = 300                  # Pre-onset extraction window (ms)
EPOCH_POST_MS = 900                 # Post-onset extraction window (ms)
BASELINE_START_MS = -200            # Pre-stimulus baseline interval start (ms)
BASELINE_END_MS = 0                 # Pre-stimulus baseline interval end (ms)
PLOT_START_MS = -200                # Plotting start time (ms)
PLOT_END_MS = 800                   # Plotting end time (ms)
HIGHLIGHT_START_MS = 200            # Analysis window start (ms) matching High-Gamma
HIGHLIGHT_END_MS = 600              # Analysis window end (ms) matching High-Gamma
SMOOTH_SIGMA = 8                    # Gaussian smoothing kernel sigma (ms)

# Visual noise level bin definitions
NOISE_BINS = {
    'Low': (0, 30),
    'Medium': (35, 65),
    'High': (70, 100)
}

# Standardized color scheme
COLORS = {
    'Hit': 'royalblue',
    'Miss': 'crimson'
}

# Category-selective electrode mappings per subject
subjects_electrodes = {
    1: [0, 25, 30, 35, 39],
    4: [23, 24],
    5: [15],
    6: [6, 9, 15, 16, 17, 23, 24, 25, 30],
    7: [8, 9, 36, 38, 39]
}


def get_noise_label(x):
    """Map continuous visual noise value into Low, Medium, or High bin label."""
    x = float(x)
    if 0 <= x <= 30:
        return 'Low'
    elif 35 <= x <= 65:
        return 'Medium'
    elif 70 <= x <= 100:
        return 'High'
    return None


def extract_beta_power(V):
    """
    Extract Beta band (13-30 Hz) power envelope from continuous signal.
    Applies 4th-order bandpass filter in SOS format, squares to obtain power,
    and extracts slow power envelope with a 3rd-order 10 Hz low-pass filter.
    """
    sos_beta = butter(
        4,
        [13, 30],
        btype='bandpass',
        fs=FS,
        output='sos'
    )
    beta_filt = sosfiltfilt(sos_beta, V, axis=0)
    beta_power = beta_filt ** 2

    sos_lp = butter(
        3,
        10,
        btype='lowpass',
        fs=FS,
        output='sos'
    )
    return sosfiltfilt(sos_lp, beta_power, axis=0)


def baseline_zscore(epoch, time_vec):
    """Standardize trial epochs against pre-stimulus baseline window (-200 to 0 ms)."""
    bl = (
        (time_vec >= BASELINE_START_MS / 1000) &
        (time_vec < BASELINE_END_MS / 1000)
    )

    mu = np.mean(epoch[:, bl], axis=1, keepdims=True)
    sd = np.std(epoch[:, bl], axis=1, keepdims=True)
    sd[sd < 1e-8] = 1e-8
    return (epoch - mu) / sd


# Store one condition-average curve per subject
subject_means = {
    noise: {'Hit': [], 'Miss': []}
    for noise in NOISE_BINS
}

trial_counts = []
time_vec = None

# Iterate across all valid subjects to compute individual condition means
for subj_id in VALID_SUBJECTS:
    print(f'\nProcessing Subject {subj_id}...')

    # dat corresponds to Experiment 2 (noisy perception task)
    dat = alldat[subj_id - 1][1]

    V = np.asarray(dat['V'], dtype=np.float32)
    t_on = np.asarray(dat['t_on']).flatten().astype(int)
    stim_cat = np.asarray(dat['stim_cat']).flatten()
    stim_noise = np.asarray(dat['stim_noise']).flatten()
    key_press = np.asarray(dat['key_press']).flatten()

    beta_power = extract_beta_power(V)

    pre = int(EPOCH_PRE_MS * FS / 1000)
    post = int(EPOCH_POST_MS * FS / 1000)
    time_vec = np.arange(-pre, post + 1) / FS

    face_chans = [
        c for c in subjects_electrodes[subj_id]
        if 0 <= c < V.shape[1]
    ]

    condition_trials = {
        noise: {'Hit': [], 'Miss': []}
        for noise in NOISE_BINS
    }

    # Extract face trials and categorize by noise level and behavioral perception
    for trial_idx, onset in enumerate(t_on):
        if stim_cat[trial_idx] != 2:
            continue

        noise_label = get_noise_label(stim_noise[trial_idx])
        if noise_label is None:
            continue

        start = onset - pre
        stop = onset + post + 1
        if start < 0 or stop > beta_power.shape[0]:
            continue

        # Behavioral response classification (Hit vs Miss)
        is_hit = np.any(
            (key_press >= onset) &
            (key_press < onset + RESPONSE_WINDOW_MS)
        )
        behavior = 'Hit' if is_hit else 'Miss'

        epoch = beta_power[start:stop, :].T
        epoch = epoch[face_chans, :]
        epoch_z = baseline_zscore(epoch, time_vec)
        roi_trace = np.mean(epoch_z, axis=0)

        condition_trials[noise_label][behavior].append(roi_trace)

    # Average trials within subject first to ensure equal subject weighting
    for noise_label in NOISE_BINS:
        for behavior in ['Hit', 'Miss']:
            trials = np.asarray(condition_trials[noise_label][behavior])

            trial_counts.append({
                'Subject': subj_id,
                'Noise': noise_label,
                'Behavior': behavior,
                'N': len(trials)
            })

            print(f'  {noise_label:6s} | {behavior:4s}: n={len(trials)}')

            if len(trials) == 0:
                continue

            # First average across trials WITHIN this subject
            subject_mean = np.mean(trials, axis=0)

            # Smooth only for visualization
            subject_mean = gaussian_filter1d(
                subject_mean,
                sigma=SMOOTH_SIGMA
            )

            subject_means[noise_label][behavior].append(subject_mean)

# ==========================================================
# GROUP PLOT: mean ± SEM across SUBJECTS
# ==========================================================
# Mask time window for plotting (-200 to +800 ms)
plot_mask = (
    (time_vec >= PLOT_START_MS / 1000) &
    (time_vec <= PLOT_END_MS / 1000)
)
plot_time_ms = time_vec[plot_mask] * 1000

fig, axes = plt.subplots(
    1, 3,
    figsize=(17, 5),
    sharex=True,
    sharey=True
)

# Plot group average curves across Low, Medium, High noise bins
for ax, noise_label in zip(axes, ['Low', 'Medium', 'High']):
    # Highlight 200-600 ms post-stimulus interval matching High-Gamma analysis
    ax.axvspan(
        HIGHLIGHT_START_MS,
        HIGHLIGHT_END_MS,
        color='gray',
        alpha=0.10,
        label='200–600 ms window'
    )

    for behavior in ['Hit', 'Miss']:
        curves = np.asarray(subject_means[noise_label][behavior])
        if len(curves) == 0:
            continue

        # Compute group-level mean and SEM across the 5 subjects
        group_mean = np.mean(curves, axis=0)

        if curves.shape[0] > 1:
            group_sem = np.std(curves, axis=0, ddof=1) / np.sqrt(curves.shape[0])
        else:
            group_sem = np.zeros_like(group_mean)

        ax.plot(
            plot_time_ms,
            group_mean[plot_mask],
            color=COLORS[behavior],
            lw=3,
            label=f'{behavior} (subjects={curves.shape[0]})'
        )

        ax.fill_between(
            plot_time_ms,
            (group_mean - group_sem)[plot_mask],
            (group_mean + group_sem)[plot_mask],
            color=COLORS[behavior],
            alpha=0.18
        )

    ax.axvline(0, color='black', linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)

    lo, hi = NOISE_BINS[noise_label]
    ax.set_title(f'{noise_label} Noise ({lo}-{hi}%)', fontweight='bold')
    ax.set_xlabel('Time (ms)')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

axes[0].set_ylabel('Beta power (baseline z-score)')

fig.suptitle(
    'Group-Average Beta Across 5 Subjects\nHit vs Miss Face Trials',
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout(rect=[0, 0, 1, 0.92])

save_name = 'Beta_GroupAverage_5Subjects_HitBlue_MissRed.png'
plt.savefig(save_name, dpi=300, bbox_inches='tight')
plt.show()

print(f'Saved: {save_name}')
