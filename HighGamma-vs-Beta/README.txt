ECoG Faces/Houses — Code Bundle
================================

These text files collect the complete analysis/plotting codes we used for Beta and High-Gamma.

Conventions used consistently:
- Sampling rate: 1000 Hz
- Face trials only for Hit/Miss analyses: stim_cat == 2
- Hit = key press in [0, 1200) ms after face onset
- Miss = no key press in that window
- Noise bins:
    Low    = 0–30
    Medium = 35–65
    High   = 70–100
- Face-selective electrodes are averaged within each subject
- Baseline: -200 to 0 ms
- Plot window: -200 to 800 ms
- Hit color: blue
- Miss color: red

Files:
1) 01_Beta_HighGamma_PerSubject_Combined.txt
   Complete combined code for Beta + High-Gamma, with one 2x3 figure per subject.

2) 02_Beta_PerSubject_Only.txt
   Beta-only version, one 1x3 figure per subject.

3) 03_HighGamma_PerSubject_Only.txt
   High-Gamma-only version, one 1x3 figure per subject.

4) 04_Beta_GroupAverage_5Subjects.txt
   Group-average Beta plot. It first computes one mean curve per Subject x Condition,
   then averages across the five subjects so every subject has equal weight.
   Perceived/Hit = blue; Non-Perceived/Miss = red.
   The 200–600 ms window is highlighted to match the High-Gamma group plot.

5) 05_HighGamma_GroupAverage_5Subjects.txt
   Group-average High-Gamma plot using equal subject weighting.
   Perceived/Hit = blue; Non-Perceived/Miss = red.
   The 200–600 ms High-Gamma analysis window is highlighted.

6) 06_HighGamma_AUC_ANOVA_200_600.txt
   High-Gamma AUC amplitude analysis in the 200–600 ms window, followed by
   repeated-measures ANOVA on Subject x Perception x Noise condition means.

Important:
- These are standalone scripts: each one loads faceshouses.npz if `alldat` is not already defined.
- Put faceshouses.npz in the current Colab working directory, or edit DATA_PATH.
- The group-average scripts do NOT pool all trials across subjects; they average within subject first.
