# ==========================================================
# HIGH-GAMMA AUC ANALYSIS + REPEATED-MEASURES ANOVA
# Pillar III: Statistical Verification of Perceptual and Noise Modulations
# Face trials only | AUC window: 200–600 ms post-stimulus onset
# Hit (Perceived) = key press in [0, 1200) ms
#
# This script computes single-trial Area Under the Curve (AUC) for High-Gamma
# broadband power in the 200–600 ms critical post-perceptual window across
# face-selective ventral temporal cortex. It then performs a 2x3 Repeated-Measures
# ANOVA (Perception [Perceived vs. Non-Perceived] x NoiseLevel [Low, Medium, High])
# using statsmodels AnovaRM to statistically validate:
#   1. Perception main effect: Significant amplitude enhancement for conscious percepts.
#   2. Noise level main effect: Parametric signal degradation with visual noise.
#   3. Perception x Noise interaction.
# ==========================================================

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from statsmodels.stats.anova import AnovaRM

# Path to Miller ECoG dataset file
DATA_PATH = 'faceshouses.npz'
if 'alldat' not in globals():
    alldat = np.load(DATA_PATH, allow_pickle=True)['dat']

# Experimental constants and analysis window definitions
FS = 1000                           # 1000 Hz acquisition rate
RESPONSE_WINDOW_MS = 1200           # Behavioral keypress reaction window [0, 1200) ms
EPOCH_PRE_MS = 300                  # Pre-onset extraction window (ms)
EPOCH_POST_MS = 900                 # Post-onset extraction window (ms)
BASELINE_WINDOW = (-0.2, 0.0)       # Pre-stimulus baseline interval (-200 to 0 ms in seconds)
AUC_WINDOW = (0.2, 0.6)             # Area Under the Curve integration window (+200 to +600 ms in seconds)

# Face-selective electrode channel assignments per subject
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


def extract_high_gamma_power(V):
    """
    Extract High-Gamma (70-150 Hz) power envelope from continuous signal.
    Applies 4th-order Butterworth bandpass filter in SOS format followed by
    Hilbert transform to obtain instantaneous analytic signal power.
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


# List to collect trial-level records across all subjects
records = []

# Iterate across all valid subjects
for subj_id, face_chans in subjects_electrodes.items():
    print('\n' + '=' * 60)
    print(f'Subject {subj_id}')
    print('=' * 60)

    # dat corresponds to Experiment 2 (noisy perception task)
    dat = alldat[subj_id - 1][1]

    V = np.asarray(dat['V'], dtype=np.float32)
    t_on = np.asarray(dat['t_on']).flatten().astype(int)
    stim_cat = np.asarray(dat['stim_cat']).flatten()
    stim_noise = np.asarray(dat['stim_noise']).flatten()
    key_press = np.asarray(dat['key_press']).flatten()

    # Extract continuous High-Gamma power
    hg_power = extract_high_gamma_power(V)

    # Build epoch time vector [-300 to +900 ms]
    pre = int(EPOCH_PRE_MS * FS / 1000)
    post = int(EPOCH_POST_MS * FS / 1000)
    time_vec = np.arange(-pre, post + 1) / FS

    # Masks for baseline and AUC integration intervals
    baseline_mask = (
        (time_vec >= BASELINE_WINDOW[0]) &
        (time_vec < BASELINE_WINDOW[1])
    )

    auc_mask = (
        (time_vec >= AUC_WINDOW[0]) &
        (time_vec <= AUC_WINDOW[1])
    )

    auc_times = time_vec[auc_mask]

    # Validate channel indices
    valid_face_chans = [
        c for c in face_chans
        if 0 <= c < V.shape[1]
    ]

    # Process each trial onset
    for trial_idx, onset in enumerate(t_on):
        # Face trials only
        if stim_cat[trial_idx] != 2:
            continue

        noise_label = get_noise_label(stim_noise[trial_idx])
        if noise_label is None:
            continue

        start = onset - pre
        stop = onset + post + 1
        if start < 0 or stop > hg_power.shape[0]:
            continue

        # Extract trial epoch, baseline standardize, and compute ROI average
        epoch = hg_power[start:stop, :].T
        epoch = epoch[valid_face_chans, :]

        bl_mean = np.mean(epoch[:, baseline_mask], axis=1, keepdims=True)
        bl_std = np.std(epoch[:, baseline_mask], axis=1, keepdims=True)
        bl_std[bl_std < 1e-8] = 1e-8
        epoch_z = (epoch - bl_mean) / bl_std

        roi_trace = np.mean(epoch_z, axis=0)
        # Compute Area Under the Curve (AUC) in the 200-600 ms window via trapezoidal integration
        auc_value = np.trapezoid(roi_trace[auc_mask], x=auc_times)

        # Behavioral detection response check
        is_hit = np.any(
            (key_press >= onset) &
            (key_press < onset + RESPONSE_WINDOW_MS)
        )

        records.append({
            'Subject': f'Subj_{subj_id}',
            'Trial': int(trial_idx),
            'Perception': 'Perceived' if is_hit else 'Non-Perceived',
            'NoiseLevel': noise_label,
            'NoiseValue': float(stim_noise[trial_idx]),
            'HG_AUC_200_600': auc_value
        })

# Trial-level table
_df = pd.DataFrame(records)
print('\nTrial-level table:', _df.shape)
display(_df.head())

# Aggregate one mean per Subject x Perception x Noise cell for repeated-measures design
# One mean per Subject x Perception x Noise cell
anova_df = (
    _df
    .groupby(['Subject', 'Perception', 'NoiseLevel'], observed=True)
    .agg(
        HG_AUC_200_600=('HG_AUC_200_600', 'mean'),
        N_Trials=('Trial', 'count')
    )
    .reset_index()
)

print('\nSubject x Condition means:')
display(anova_df)

# Repeated-measures ANOVA (2x3: Perception x NoiseLevel within Subject)
# Repeated-measures ANOVA
anova_hg_auc = AnovaRM(
    data=anova_df,
    depvar='HG_AUC_200_600',
    subject='Subject',
    within=['Perception', 'NoiseLevel']
).fit()

print('\n' + '=' * 65)
print('ANOVA: HIGH-GAMMA AUC — 200–600 ms')
print('=' * 65)
print(anova_hg_auc)

# Save statistical summary tables to CSV
# Save tables
_df.to_csv('HighGamma_AUC_TrialLevel_200_600.csv', index=False)
anova_df.to_csv('HighGamma_AUC_SubjectConditionMeans_200_600.csv', index=False)

print('\nSaved:')
print('HighGamma_AUC_TrialLevel_200_600.csv')
print('HighGamma_AUC_SubjectConditionMeans_200_600.csv')
