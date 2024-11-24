# CDFM

## Introduction
Non-stationarity is an intrinsic property of real-world time series data and plays a crucial role in time series forecasting. However, recent methods that directly apply normalization to input data may result in the loss of essential statistical information, leading to three issues: (1) disrupting global temporal dependencies, (2) ignoring channel-specific differences, and (3) producing over-smoothed predictions.
To address these issues, we first provide a theoretical proof demonstrating the positive correlation between non-stationarity and variance.
Based on the analysis, we propose a novel lightweight Channel-wise Dynamic Fusion Model (CDFM), which selectively and dynamically recovers intrinsic non-stationarity in raw series while still keeping the predictability of normalized series.

![framework](fig/framework.png)

We conduct extensive experiments to evaluate the performance and efficiency of our model on seven widely-used real-world time series datasets.

Multivariate forecasting results:

![predictions](fig/predictions.png)

## Usage

### Data Preparation

All the 7 datasets are available at the [Google Driver](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided by Autoformer.


### Training Example

We provide scripts for CDFM. You can easily reproduce the results from the paper by running the provided script command.

```
sh scripts/CDFM/etth1.sh
```

## Acknowledgement
We gratefully acknowledge the following GitHub repositories for their valuable codebases and datasets:
- DLinear (https://github.com/cure-lab/LTSF-Linear)
- PatchTST (https://github.com/yuqinie98/patchtst)