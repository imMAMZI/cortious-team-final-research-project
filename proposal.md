# dataset information:

We will use the Neuromatch ECoG Faces/Houses dataset, which contains intracranial ECoG recordings from 7 human participants during face perception tasks. The dataset includes:

- Face–House localizer task: used to identify face-selective electrodes.
- Noisy face perception task: face stimuli presented with 0–100% visual noise.
- Neural data: high-temporal-resolution ECoG signals.
- Behavioral data: participants pressed a key when they perceived the face, allowing us to classify trials as perceived or non-perceived.

# 1. What are your main research questions?

Write one to three specific questions that you can realistically investigate using your dataset.

main question

Investigating Neural Response Amplitude and Latency Differences Between Perceived and Non-Perceived Face Trials Under Different Noise Levels

1) How do high-gamma and broadband response amplitudes differ between perceived and non-perceived face trials under different levels of visual noise?

2)Does visual noise affect the timing of high-gamma and broadband neural responses during face perception?

3)Can high-gamma and broadband neural activity distinguish between perceived and non-perceived face trials across different noise levels?

# 2. Why are these questions interesting or important?

Briefly explain why answering them would be scientifically useful. 
Face perception allows us to recognize faces even when visual information is degraded by noise. Previous ECoG studies have linked high-gamma and broadband activity to visual processing, but it remains unclear how these responses differ between perceived and non-perceived faces under different noise levels. This project will examine whether these differences are reflected in neural response amplitude, latency, or both, and whether high-gamma and broadband activity can distinguish between perceived and non-perceived face trials.

# 3. What is already known about these questions?

Identify two or three relevant papers and briefly explain what each contributes to your project. 

Prior ECoG research has established event-related broadband (ERBB) activity, broadband electrocorticographic responses, and high-gamma activity as important neural indices of face processing in the ventral temporal cortex. Miller et al. (2016) demonstrated that ERBB and ERP responses capture complementary aspects of perceptual states, and their combination improves prediction of stimulus onset and category. Miller et al. (2017) further showed that broadband ECoG responses reflect hierarchical perceptual processing, with posterior fusiform regions exhibiting graded, noise-sensitive responses to faces, anterior fusiform regions showing more binary responses, and pericalcarine cortex displaying earlier but category-nonselective activity. Additionally, Miller et al. (2015) demonstrated that both neural response magnitude and latency are influenced by stimulus context, with novel face stimuli eliciting stronger responses and shorter latencies compared with repeated faces at face-selective sites. More recently, Li et al. (2021) showed that increasing visual noise reduces both face-recognition accuracy and high-gamma power, with HGP magnitude positively correlated with recognition performance

# 4. What could your project confirm or add to the existing literature?

Based on the background above, explain how your project might confirm, challenge, extend, or connect previous findings.

Our project aims to investigate whether conscious visual perception is better indexed by both response magnitude (amplitude) and processing speed/dynamics (latency), rather than by signal power alone. Additionally, our project extends previous research by directly comparing perceived and non-perceived face trials under different levels of visual noise. While previous studies have mainly focused on the relationship between neural response magnitude and face-recognition performance, our project further investigates neural response latency and compares high-gamma and broadband activity. Furthermore, we examine whether these neural features can distinguish between perceived and non-perceived trials across different noise levels.

# 5. Why is this dataset suitable for answering your research questions?

Explain how the available measurements and variables relate to your questions. Mention any important information that is missing

This dataset provides high-resolution ECoG voltage data, allowing us to extract precise high-gamma and broadband amplitude and latency metrics. Its variables directly address our hypotheses: stim_noise tracks visual degradation, while key_press serves as a behavioral proxy for subjective perception. Furthermore, the noise-free localizer task allows for independent identification of face-selective electrodes. However, the lack of eye-tracking data means we cannot definitively rule out inattention or blinking as the cause for "non-perceived" trials.

# 6. What information from the dataset will you use, and what have you learned from exploring it so far?

Briefly describe which observations, variables, or experimental conditions you will examine. Include at least one preliminary plot, observation, or practical issue.

We will use voltage, stimulus category, and onset times from the localizer task to identify face-selective channels, then extract high-frequency features from the noisy task based on noise level and key presses. A major practical issue is that two subjects lack key-press data and must be excluded from our behavioral analysis. Preliminary exploration reveals we must target the 40-60% visual noise threshold, as this provides a necessary balance of perceived and non-perceived trials at identical noise levels. Finally, extracting reliable single-trial peak latency requires aggressive low-pass smoothing to handle inherent ECoG signal noise.

# 7. What are your main hypotheses?

State the results or relationships you expect to observe. If possible, make your predictions directional and measurable.

H1 – Neural response amplitude: Perceived face trials will show significantly greater high-gamma and broadband response amplitudes than non-perceived trials, particularly at intermediate noise levels (approximately 40–60%). Increasing visual noise is expected to progressively reduce response amplitude.
H2 – Neural response latency: Non-perceived face trials will show delayed high-gamma and broadband responses compared with perceived trials. Increasing visual noise is expected to increase response latency, reflecting slower neural processing.
H3 – Neural decoding of perception: High-gamma and broadband activity will distinguish perceived from non-perceived face trials above chance level. We expect combining amplitude and latency features to provide better classification than either feature alone, with high-gamma potentially providing greater sensitivity to perceptual differences.

# 8. How will you test your hypotheses?

Describe the simplest possible analysis or modeling workflow, from selecting and preparing the data to producing the results. 
We will preprocess the ECoG signals using gamma band filtering and Morlet wavelets, following the approach of Miller et al. We will compare perceived (button press) and non-perceived trials at each noise level and examine broadband and high gamma response magnitude and latency at face selective electrodes. A 2 × 3 ANOVA will test the effects of perception and noise level, while LDA will test whether neural activity can distinguish perceived from non-perceived trials.

# 9. How will you evaluate and interpret the results?

What plots, statistics, performance measures, baselines, or comparisons will tell you whether the evidence supports your hypotheses? 
We will evaluate the results using ANOVA statistics (p < 0.05), post hoc comparisons, and linear discriminant analysis (LDA). We will visualize neural responses over time and compare response magnitude and latency across perceived and non perceived trials and different noise levels.

# 10. What do you think the final output of your project might look like?

Describe some of the main figure, table, model, or other result you currently expect to produce and explain how it would help answer your research questions. This is a preliminary idea and may change as the project develops. 

We expect to produce plots of response magnitude and latency across noise levels, comparing broadband and high-gamma activity between perceived and non-perceived face trials at face-selective electrodes. Differences will be tested using ANOVA with post hoc comparisons, and LDA will assess how well these features separate perceived from non-perceived trials.
