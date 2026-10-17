# Neural Signatures of Face Perception in Human ECoG Under Matched Visual Evidence

## Abstract

The neural mechanisms underlying the extraction of facial information under visually degraded conditions remain a fundamental question in computational neuroscience. This project investigates the spatiotemporal dynamics of human visual perception utilizing a high-resolution electrocorticographic (ECoG) dataset (Miller, 2019). By analyzing recordings from the ventral temporal cortex of human participants performing a phase-scrambled face/house detection task, we decode the occurrence, timing, and content of conscious perception. We establish three analytical pillars: Event-Related Potentials (ERPs), Linear Discriminant Analysis (LDA) decoding, and Spectral Dynamics (High-Gamma vs. Beta bands). Our results demonstrate that high-gamma broadband activity (a surrogate for asynchronous population firing) and ERPs (reflecting synchronized feed-forward activation) provide complementary physiological information. Together, they reveal that conscious face perception is characterized by distinct amplitude enhancements and latency shifts that degrade parametrically with visual noise, ultimately allowing for the spontaneous decoding of perceptual states.

## Dataset Architecture

The dataset comprises intracranial ECoG recordings from 7 human epilepsy patients sampled at 1000 Hz. The experimental paradigm consists of two phases:

1. **Face–House Localizer Task:** A noise-free passive viewing task used to independently identify category-selective (face/house) electrodes in the ventral temporal cortex.
2. **Noisy Face Perception Task:** Phase-scrambled images of faces and houses presented for 1000 ms with visual noise ranging from 0% to 100% (in 5% increments). Participants pressed a key when they consciously perceived a face, providing a direct behavioral proxy for subjective perception (Hit vs. Miss).

Signal preprocessing includes common average referencing (CAR), 3rd-order Butterworth notch filtering for line noise (58–62 Hz, 118–122 Hz, 178–182 Hz), and z-scoring relative to a pre-stimulus baseline (-200 to 0 ms).

---

## Pillar I: Event-Related Potentials (ERP) Dynamics

### Theoretical Background

ERPs in the ventral temporal lobe, particularly the face-selective "N200" or "N170" components, reflect the initial, synchronized feed-forward sweep of neural activation from early visual areas. While highly polymorphic across adjacent cortical sites, ERPs provide exceptional temporal precision regarding stimulus onset. Analyzing ERPs under varying noise conditions allows us to isolate the accumulation of sensory evidence prior to top-down categorical assignment.

### Methodology

Raw voltage traces were high-pass filtered at 0.5 Hz to remove slow drift, epoch-locked to stimulus onset (-200 to 500 ms), and baseline-corrected. Trials were categorized by behavioral response (Perceived vs. Not Perceived) and stratified across three custom noise bins: Low (0–30%), Medium (35–65%), and High (70–100%).

![Ventral Temporal ERP](https://github.com/imMAMZI/cortious-team-final-research-project/blob/main/ERP/ventral-temporal-erp.jpg)
**Observations:** This plot illustrates the grand-average ERP for face trials, grouped by behavioral outcome (Perceived vs. Not Perceived) across all noise levels. A pronounced negative deflection is observed within the 200–280 ms window for perceived faces (blue), within the latency range typically associated with face-sensitive N170/N200 responses. In comparison, trials in which participants did not report perceiving a face (red) show a noticeably attenuated response. This pattern suggests that ERP amplitude in this time window may be associated with behavioral reports of face perception, beyond the mere presentation of a face stimulus.

![Grand Average ERP Across Noise Bins](https://github.com/imMAMZI/cortious-team-final-research-project/blob/main/ERP/grand-average-ventral-temporal-erp.jpg)
**Observations:** By stratifying the ERPs across visual noise bins, we observe the parametric degradation of the feed-forward signal. In the Low Noise (0-30%) bin, the divergence between Perceived and Not Perceived trials is massive. In the Medium Noise (35-65%) bin—approaching the psychophysical threshold—the N170 amplitude is significantly blunted, yet a discernible difference between Hits and Misses remains. In the High Noise (70-100%) bin, the signal collapses, reflecting a failure of sensory evidence accumulation, aligning with the behavioral inability to detect faces.

![Balanced Grand Average ERP](https://github.com/imMAMZI/cortious-team-final-research-project/blob/main/ERP/balanced-grand-average-erp.jpg)
**Observations:** To ensure the amplitude differences observed above were not statistical artifacts of imbalanced trial counts (e.g., very few "Misses" in Low Noise, very few "Hits" in High Noise), this plot utilizes strict within-subject, within-bin random subsampling to equalize trial counts. The preservation of the N170 amplitude divergence in the Low and Medium noise bins confirms that the neural signature of conscious perception is robust and independent of sample size disparities.

---

## Pillar II: Linear Discriminant Analysis (LDA) Decoding

### Theoretical Background

To spontaneously decode perceptual states, we project high-dimensional ECoG features into a unified decision space. Previous literature establishes that the physiological threshold of category-selective ventral temporal loci perfectly mirrors the behavioral perceptual threshold. We utilize Fisher Linear Discriminant Analysis (LDA) to test whether neural activity can distinguish perceived from non-perceived trials across noise levels.

### Classification Pipeline

We implemented a 5-fold Stratified Cross-Validation LDA using the `lsqr` solver with automatic shrinkage. Decoding was performed in two ways: 

1. **Static Window:** Averaging features over a predefined physiological window (e.g., 200-280 ms for ERP, 100-400 ms for Broadband).
2. **Time-Resolved:** A sliding 20 ms window to track decoding accuracy over the trial timecourse.

![Whole Data Static LDA](https://github.com/imMAMZI/cortious-team-final-research-project/blob/main/LDA/whole_data_3bins_static_lda.png)
**Observations:** This plot displays the static LDA decoding accuracy using raw voltage features across all electrodes. Decoding accuracy drops parametrically from ~65-70% in the Low Noise bin to theoretical chance (50%) in the High Noise bin. The Grand Average (crimson) perfectly mirrors the behavioral psychometric curve, demonstrating that the linear separability of the raw voltage brain state degrades proportionally with visual noise.

![Whole Data Time-Resolved LDA](https://github.com/imMAMZI/cortious-team-final-research-project/blob/main/LDA/whole_data_3bins_time_resolved_lda.png)
**Observations:** The time-resolved LDA on raw voltage reveals the precise temporal dynamics of perceptual decoding. For Low Noise (blue), decoding accuracy sharply peaks at ~68% exactly within the N170 window (200-280 ms), reflecting the synchronized feed-forward sweep. Medium (orange) and High (red) noise levels fail to achieve significant decoding accuracy at any time point, indicating that visual degradation destroys the temporal phase-locking required for voltage-based classification.

![Broadband Perceived Static LDA](https://github.com/imMAMZI/cortious-team-final-research-project/blob/main/LDA/broadband_perceived_static_lda.png)
**Observations:** Shifting the feature space to True Broadband Power (70-150 Hz), this static LDA plot evaluates decoding accuracy over the 100-400 ms window. Broadband power, which reflects asynchronous population firing, provides a more robust and stable feature space for classification. The degradation of accuracy from Low to High noise is smoother, suggesting that broadband activity captures a more integrated, recurrent level of perceptual processing compared to the transient ERP.

![Broadband Perceived Time-Resolved LDA](https://github.com/imMAMZI/cortious-team-final-research-project/blob/main/LDA/broadband_perceived_time_resolved_lda.png)
**Observations:** The time-resolved broadband LDA shows a fundamentally different temporal profile than the raw voltage LDA. Instead of a sharp, transient peak, the Low Noise decoding accuracy rises at ~150 ms and remains sustained well beyond 400 ms. This sustained decoding accuracy supports the hypothesis that high-gamma broadband activity reflects continuous, recurrent cortico-cortical processing and active maintenance of the face percept, rather than a fleeting feed-forward trigger.

---

## Pillar III: Spectral Dynamics (High-Gamma vs. Beta)

### Theoretical Background

Cortical spectral dynamics offer distinct physiological windows into brain function. High-Gamma (60–160 Hz) broadband activity is a reliable surrogate for asynchronous local neuronal population firing rates. Conversely, lower frequency bands like Beta (13–30 Hz) are typically associated with synchronous, modulatory, or state-based processes. We hypothesize that High-Gamma will tightly track conscious perception (Hits vs. Misses), while Beta will exhibit different, potentially inverse or purely transient dynamics.

### Signal Processing

- **High-Gamma (70–150 Hz):** Extracted using a 4th-order Butterworth bandpass filter followed by a Hilbert transform to obtain the analytic signal amplitude (power envelope).
- **Beta (13–30 Hz):** Extracted via bandpass filtering, squared to obtain raw power, and subsequently smoothed with a 10 Hz low-pass filter to extract the slow power envelope.
- **Statistical Testing:** A 2x3 Repeated-Measures ANOVA (Perception x NoiseLevel) was applied to the Area Under the Curve (AUC) of the High-Gamma power within the 200–600 ms window.

![Grand Average High-Gamma](https://github.com/imMAMZI/cortious-team-final-research-project/blob/main/HighGamma-vs-Beta/grand-average-high-gamma-across-all-subjects.jpg)
**Observations:** This figure highlights the profound sensitivity of High-Gamma power to conscious perception. In the Low Noise bin, Perceived trials (blue) elicit a massive, sustained increase in High-Gamma power starting at ~200 ms and lasting until ~600 ms. Non-perceived trials (red) show a significantly blunted response. In the Medium Noise bin, the amplitude of the Perceived trials drops, but the divergence between Hit and Miss remains clear. In the High Noise bin, the signal fails to rise above baseline, confirming that High-Gamma power is an all-or-nothing index of successful face percept formation in the ventral temporal cortex.

![Grand Average Beta](https://github.com/imMAMZI/cortious-team-final-research-project/blob/main/HighGamma-vs-Beta/grand-average-beta-dynamics-across-all-subjects.jpg)
**Observations:** The Beta band dynamics present a striking contrast to High-Gamma. Instead of tracking successful perception with sustained power increases, Beta power exhibits a transient, sharp spike around 200-300 ms that is actually *higher* for Non-perceived trials (red) in the Low Noise condition. This suggests that Beta may reflect a synchronous inhibitory gating mechanism, prediction error, or a state of heightened effort/ambiguity when the visual system fails to resolve a clear percept despite low visual noise. It does not serve as a direct surrogate for active stimulus representation.

---

## Installation & Reproducibility

To reproduce the signal processing, LDA classification, and visualizations, ensure your environment meets the following requirements:

### Prerequisites

- Python 3.8+
- `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `statsmodels`

### Setup Instructions

1. **Clone the repository:**
   
   ```bash
   git clone https://github.com/your-repo/ecog-visual-decoding.git
   cd ecog-visual-decoding
   ```

2. **Install dependencies:**
   
   ```bash
   pip install -r requirements.txt
   ```

### Data Placement

Ensure the Miller ECoG dataset file (`faceshouses.npz`) is placed in the root directory of the project.

### Execute Analytical Scripts

Run the scripts in the following order to generate the respective analyses:

```bash
python erp.py                                 # Generates ERP plots
python whole_data_LDA_3bins.py                # Generates Voltage LDA plots
python corrected_broadband_LDA.py             # Generates Broadband LDA plots
python 05_HighGamma_GroupAverage_5Subjects.py # Generates High-Gamma dynamics
python 04_Beta_GroupAverage_5Subjects.py      # Generates Beta dynamics
python 06_HighGamma_AUC_ANOVA_200_600.py      # Runs RM-ANOVA statistics
```

---

## Interpretation of Visual Data

The synthesis of our three analytical pillars provides a comprehensive view of how the human ventral temporal lobe constructs visual percepts:

* **Amplitude Correlates with Perception (H1):** Both the ERP (N170) and High-Gamma broadband analyses confirm that neural response amplitude is significantly greater for perceived faces compared to non-perceived faces. The High-Gamma signal, in particular, acts as a sustained physiological signature of conscious percept formation.
* **Latency and Temporal Dynamics (H2):** Time-resolved LDA and continuous spectral plots reveal that visual noise not only blunts amplitude but delays and smears the neural response. The sharp temporal precision of the ERP is lost as noise increases, reflecting slower, struggling neural processing.
* **Complementary Decoding (H3):** The LDA classification proves that neural activity can distinguish perceived from non-perceived trials well above chance. Crucially, the transient nature of the ERP decoding (peaking sharply at 200 ms) versus the sustained nature of the Broadband decoding (100–400+ ms) proves that these two signals capture complementary physiological phenomena: the initial synchronized feed-forward visual evidence (ERP) and the subsequent asynchronous recurrent processing required for categorical assignment and conscious awareness (High-Gamma).

---

## Contact us on Linkedin!

| Name                  | Linkedin                                                     |
| --------------------- | ------------------------------------------------------------ |
| Mohammadreza Khashei  | https://www.linkedin.com/in/immamziii/                       |
| Sama Arjaneh          | https://www.linkedin.com/in/sama-arjeneh-7b680642a/          |
| Negar Gorji           | https://www.linkedin.com/in/negar-k-gorji-777438248/         |
| Mahtab Mohammadi      | https://www.linkedin.com/in/mahtab-mohammadii/               |
| Sajedeh Saberi        | https://www.linkedin.com/in/sajedehsaberi/                   |
| Monireh Falahat       | https://www.linkedin.com/in/monireh-falahat-924a62355/       |
| Asal Shirvan          | -                                                            |
| FatemeZahra Habibpour | https://www.linkedin.com/in/fatemezahra-habibpour-4b2b51431/ |
