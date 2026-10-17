# =========================================================================
# Pillar III: Spectral Dynamics — Beta (13–30 Hz) vs. High-Gamma (70–150 Hz)
# Per-Subject Combined Analysis (Face Trials Only)
#
# Hit (Perceived) = Blue | Miss (Unperceived) = Red
#
# This script computes and compares the temporal power dynamics of two distinct
# frequency bands within face-selective ventral temporal cortex across 3 noise bins:
#   - Beta (13–30 Hz): Modulatory/inhibitory state indicator, extracted via bandpass,
#     squaring, and 10 Hz lowpass filtering.
#   - High-Gamma (70–150 Hz): Surrogate for asynchronous local population firing,
#     extracted via bandpass filtering and Hilbert analytic power envelope.
#
# Generates a 2x3 grid per subject (Row 1: Beta, Row 2: High-Gamma; Cols: Low, Med, High noise).
# =========================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import gaussian_filter1d

# =========================================================
# 1) LOAD DATA
# =========================================================
# Path to Miller ECoG dataset file
DATA_PATH = 'faceshouses.npz'

# Load alldat if not already present in memory
if 'alldat' not in globals():
    alldat = np.load(DATA_PATH, allow_pickle=True)['dat']

# =========================================================
# 2) SETTINGS & HYPERPARAMETERS
# =========================================================

FS = 1000                   # Acquisition sampling rate in Hz
RESPONSE_WINDOW_MS = 1200   # Behavioral keypress response window post-stimulus onset [0, 1200) ms

# Use a slightly longer extraction epoch, but only display -200 to +800 ms
EPOCH_PRE_MS = 300          # Pre-stimulus epoch extraction duration (ms)
EPOCH_POST_MS = 900         # Post-stimulus epoch extraction duration (ms)
PLOT_START_MS = -200        # Visualization window start (ms)
PLOT_END_MS = 800           # Visualization window end (ms)

BASELINE_START_MS = -200    # Pre-stimulus baseline interval start (ms)
BASELINE_END_MS = 0         # Pre-stimulus baseline interval end (ms)
SMOOTH_SIGMA = 8            # 1D Gaussian smoothing kernel standard deviation (ms) for plotting

# Noise level grouping definitions
NOISE_BINS = {
    'Low': (0, 30),
    'Medium': (35, 65),
    'High': (70, 100)
}

# Frequency band extraction configuration
BANDS = {
    'Beta': {
        'range': (13, 30),
        'method': 'squared'
    },
    'High-Gamma': {
        'range': (70, 150),
        'method': 'hilbert'
    }
}

# Face-selective electrode channel IDs localized in ventral temporal cortex
subjects_electrodes = {
    1: [0, 25, 30, 35, 39],
    4: [23, 24],
    5: [15],
    6: [6, 9, 15, 16, 17, 23, 24, 25, 30],
    7: [8, 9, 36, 38, 39]
}

# Output directory for saving generated per-subject figures
SAVE_DIR = 'Beta_HG_PerSubject_Plots'
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# 3) FUNCTIONS
# =========================================================

def extract_band_power(v_continuous, fs, lowcut, highcut, method='hilbert'):
    """
    Continuous band-power extraction before epoching.
    
    Applies a 4th-order Butterworth bandpass filter in second-order sections (SOS)
    to minimize numerical instability, followed by envelope extraction:
      - 'hilbert': Absolute squared analytic signal (for High-Gamma broadband).
      - 'squared': Squared filtered signal smoothed with 10 Hz low-pass filter (for Beta).
    """
    sos = butter(
        4,
        [lowcut, highcut],
        btype='bandpass',
        fs=fs,
        output='sos'
    )

    filtered = sosfiltfilt(sos, v_continuous, axis=0)

    if method == 'hilbert':
        analytic = hilbert(filtered, axis=0)
        power = np.abs(analytic) ** 2

    elif method == 'squared':
        power = filtered ** 2
        sos_lp = butter(
            3,
            10,
            btype='lowpass',
            fs=fs,
            output='sos'
        )
        power = sosfiltfilt(sos_lp, power, axis=0)

    else:
        raise ValueError("method must be 'hilbert' or 'squared'")

    return power


def classify_trials(dat, response_window_ms=1200):
    """
    Hit if key press occurs in [0, 1200) ms after stimulus onset.
    Returns boolean array indicating behavioral detection response per trial.
    """
    t_on = np.asarray(dat['t_on']).flatten().astype(int)
    keypresses = np.asarray(dat['key_press']).flatten()

    trial_has_response = np.zeros(len(t_on), dtype=bool)

    for i, onset in enumerate(t_on):
        trial_has_response[i] = np.any(
            (keypresses >= onset) &
            (keypresses < onset + response_window_ms)
        )

    return trial_has_response


def epoch_continuous_signal(continuous_data, t_on):
    """
    Epoch continuous power data into 3D array (trials x channels x time)
    locked to stimulus onset timestamps.
    """
    sample_pre = int(EPOCH_PRE_MS * FS / 1000)
    sample_post = int(EPOCH_POST_MS * FS / 1000)

    time_vec = np.arange(-sample_pre, sample_post + 1) / FS

    epochs = []
    valid_indices = []

    for i, onset in enumerate(t_on):
        start = onset - sample_pre
        end = onset + sample_post + 1

        if start >= 0 and end <= continuous_data.shape[0]:
            epochs.append(continuous_data[start:end, :])
            valid_indices.append(i)

    epochs = np.asarray(epochs)                 # trials x time x channels
    epochs = np.transpose(epochs, (0, 2, 1))  # trials x channels x time

    return epochs, np.asarray(valid_indices), time_vec


def baseline_zscore(epochs, time_vec):
    """
    Standardize epoched power relative to pre-stimulus baseline (-200 to 0 ms).
    Computes mean and std across the baseline window per trial and channel.
    """
    baseline_mask = (
        (time_vec >= BASELINE_START_MS / 1000) &
        (time_vec < BASELINE_END_MS / 1000)
    )

    baseline_mean = np.mean(
        epochs[:, :, baseline_mask],
        axis=-1,
        keepdims=True
    )

    baseline_std = np.std(
        epochs[:, :, baseline_mask],
        axis=-1,
        keepdims=True
    )

    baseline_std = np.where(baseline_std < 1e-8, np.nan, baseline_std)
    return (epochs - baseline_mean) / baseline_std


def analyze_subject(subj_id):
    """
    Extract, epoch, baseline standardize, and ROI-average Beta and High-Gamma power
    for an individual subject during face trials.
    """
    # dat corresponds to Experiment 2 (noisy perception task)
    dat = alldat[subj_id - 1][1]

    # Skip subjects lacking behavioral keypress data
    if 'key_press' not in dat or dat['key_press'] is None:
        print(f'Subject {subj_id}: no behavioral data')
        return None

    V = np.asarray(dat['V'], dtype=np.float32)
    t_on = np.asarray(dat['t_on']).flatten().astype(int)
    stim_cat = np.asarray(dat['stim_cat']).flatten()
    stim_noise = np.asarray(dat['stim_noise']).flatten()

    trial_response = classify_trials(dat, RESPONSE_WINDOW_MS)

    # Filter for valid face-selective electrodes present in this subject's recording
    chans = [
        c for c in subjects_electrodes[subj_id]
        if 0 <= c < V.shape[1]
    ]

    if len(chans) == 0:
        print(f'Subject {subj_id}: no valid face-selective electrodes')
        return None

    band_results = {}

    # Extract continuous power and epoch for each target frequency band
    for band_name, config in BANDS.items():
        lowcut, highcut = config['range']
        method = config['method']

        # Extract continuous power envelope across all electrodes
        continuous_power = extract_band_power(
            V,
            FS,
            lowcut,
            highcut,
            method=method
        )

        # Epoch into trials
        epochs, valid_idx, time_vec = epoch_continuous_signal(
            continuous_power,
            t_on
        )

        # Apply trial-by-trial baseline z-scoring
        epochs_z = baseline_zscore(epochs, time_vec)

        # Average across face-selective electrodes (Region of Interest - ROI)
        roi_power = np.nanmean(
            epochs_z[:, chans, :],
            axis=1
        )

        # Smooth temporal profile with 1D Gaussian kernel for display
        roi_power_smooth = gaussian_filter1d(
            roi_power,
            sigma=SMOOTH_SIGMA,
            axis=-1
        )

        band_results[band_name] = {
            'traces': roi_power_smooth,
            'valid_idx': valid_idx,
            'time_vec': time_vec
        }

    return {
        'band_results': band_results,
        'stim_cat': stim_cat,
        'stim_noise': stim_noise,
        'trial_response': trial_response
    }


def plot_subject_beta_hg(subj_id, analysis_result):
    """
    Generate and save a 2x3 panel plot comparing Beta vs High-Gamma dynamics
    (Hit in Blue vs Miss in Red) across Low, Medium, and High noise bins.
    """
    band_results = analysis_result['band_results']
    stim_cat = analysis_result['stim_cat']
    stim_noise = analysis_result['stim_noise']
    trial_response = analysis_result['trial_response']

    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 8),
        sharex=True,
        sharey='row'
    )

    band_order = ['Beta', 'High-Gamma']

    # Iterate through frequency bands (Rows) and noise bins (Columns)
    for row, band_name in enumerate(band_order):
        result = band_results[band_name]

        traces = result['traces']
        valid_idx = result['valid_idx']
        time_vec = result['time_vec']

        noise_all = stim_noise[valid_idx]
        face_all = stim_cat[valid_idx] == 2
        resp_all = trial_response[valid_idx]

        for col, (noise_label, (low, high)) in enumerate(NOISE_BINS.items()):
            ax = axes[row, col]

            # Mask face trials belonging to current noise bin
            noise_mask = (noise_all >= low) & (noise_all <= high)
            hit_mask = face_all & noise_mask & resp_all
            miss_mask = face_all & noise_mask & (~resp_all)

            # Plot Hit and Miss condition traces with SEM bands
            for mask, label, color in [
                (hit_mask, 'Hit', 'royalblue'),
                (miss_mask, 'Miss', 'crimson')
            ]:
                n = np.sum(mask)
                if n == 0:
                    continue

                data = traces[mask]
                mean_trace = np.nanmean(data, axis=0)

                if n > 1:
                    sem_trace = np.nanstd(data, axis=0, ddof=1) / np.sqrt(n)
                else:
                    sem_trace = None

                ax.plot(
                    time_vec * 1000,
                    mean_trace,
                    color=color,
                    lw=2,
                    label=f'{label} (n={n})'
                )

                if sem_trace is not None:
                    ax.fill_between(
                        time_vec * 1000,
                        mean_trace - sem_trace,
                        mean_trace + sem_trace,
                        color=color,
                        alpha=0.22
                    )

            # Formatting and reference markings
            ax.axvline(0, color='black', linestyle='--', alpha=0.6)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
            ax.set_xlim(PLOT_START_MS, PLOT_END_MS)
            ax.grid(True, alpha=0.25)

            if row == 0:
                ax.set_title(
                    f'{noise_label} Noise\n({low}-{high}%)',
                    fontsize=13,
                    fontweight='bold'
                )

            if col == 0:
                ax.set_ylabel(f'{band_name}\nZ-scored power', fontsize=13)

            if row == 1:
                ax.set_xlabel('Time (ms)', fontsize=12)

            ax.legend(fontsize=9, loc='upper right')

    plt.suptitle(
        f'Subject {subj_id} — Beta & High-Gamma Dynamics',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    save_path = os.path.join(
        SAVE_DIR,
        f'Subject_{subj_id}_Beta_HG_HitBlue_MissRed.png'
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f'Saved: {save_path}')

# =========================================================
# 4) RUN ALL SUBJECTS
# =========================================================
# Execute analysis and figure generation for all valid subjects
for subj_id in subjects_electrodes.keys():
    print('\n' + '=' * 60)
    print(f'Subject {subj_id}')
    print('=' * 60)

    result = analyze_subject(subj_id)

    if result is None:
        continue

    plot_subject_beta_hg(subj_id, result)

print('\nDone.')
print(f'Figures saved in: {SAVE_DIR}')
