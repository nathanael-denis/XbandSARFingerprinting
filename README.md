# X-band Synthetic Aperture Radar Satellites Fingerprinting

An authentication pipeline for Synthetic Aperture Radar (SAR) satellites utilizing grayscale spectrogram classification and contrastive learning techniques.

---

## Overview

This repository provides a comprehensive framework for authenticating SAR satellites by converting In-phase and Quadrature (IQ) data into spectrograms and employing machine learning models for classification. The pipeline incorporates preprocessing steps, contrastive learning for feature extraction, and evaluation metrics to assess model performance.

---


## Motivation

The novelty of the proposed methodology is to fingerprint (and, by extension, authenticate satellites with physical-layer data) satellites operating in the X-band. For the ICEYE constellation, it is not covered by the existing satellite fingerprinting literature (to the best of our knowledge), as it is composed of radar satellites and also operates outside the typical software-defined radio range, and therefore requires dedicated hardware and software. After a data collection campaign spanning over 4 weeks, we demonstrate that with grayscale spectrograms derived from IQ data, it is possible to create a fingerprint for a specific satellite. This fingerprint can further be used for physical-layer authentication.

---
## Threat model

SAR satellites are monostatic; they receive their own illuminating signals and can infer data from them. Theoretically, they can self-authenticate their signals to avoid receiving crafted radar signals from the ground and create maliciously modified images. This is NOT the model considered here. Instead, we use the illuminating signals collected from the ground and use them to authenticate the satellite later on. This assumes that the satellite is illuminating the ground station area using its radar when it downlinks the data. 

## Repository Structure

- **Documents/**: Contains radio pass information.  
- **Forecast/**: Scripts to predict radio passes for the next 3 days, tuned to ICEYE but can be adapted to any satellite with a Norad ID.
- **Plots/**: Creates the figures for the article.  
- **Results/**: Stores results from experiments and evaluations, in Excel sheets as well as output from Slurm (HPC management software).  
- **BalancedSplit.py**: Script for creating balanced datasets by ensuring equal representation of classes (not used for the experiments).  
- **ContrastiveLearning.py**: Implementation of contrastive learning techniques for feature extraction (used for the SCULLY methodology).  
- **IQtoSpectrograms.py**: Converts raw IQ data into grayscale spectrograms.  
- **OneVsRest.py**: Script for implementing one-vs-rest classification strategy (used for the MULDER methodology).  
- **Prefiltering.py**: Applies magnitude-based filtering to raw data to remove noise and reduce the massive volume of data.  
- **SNRperSample.py**: Calculates Signal-to-Noise Ratio for a given file.  Used to generate a figure in the article.
- **UnbalancedSplit.py**: Script for creating unbalanced datasets, keeping the natural imbalances in satellite passes (used for the experiments). F1-score is preferred over accuracy to handle class imbalances.  
- **confusion_matrix_40103630.csv**: CSV file containing confusion matrix data for model evaluation (as an example).  

---

## Installation

Ensure you have Python 3.8 or higher installed. Then, install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

To run the experiments, execute the scripts in the following order:

[   Prefilter --> Convert to spectrograms --> Create datasets by splitting data --> Train the classifier --> Testing  ]

### 1. Apply Preprocessing Steps

Fetch the data from https://drive.proton.me/urls/470HH26X9G#MCHdOH2JkSXS and place it into RawIQ/. Then run:
```bash
python Prefiltering.py
```
This will remove noise from the IQ samples based on a magnitude-based threshold.

### 2. Convert IQ Samples to Spectrograms
```bash
python IQtoSpectrograms.py
```
This generates images for each satellite under folders named X1, ..., X51.

### 3. Dataset Splitting

Create balanced or unbalanced datasets:
```bash
python BalancedSplit.py
python UnbalancedSplit.py
```
- Balanced datasets have the same number of images per class (use other metrics if you choose this option).
- Unbalanced datasets are used in the experiments described in the paper and rely on F1-score for evaluation.

### 4. Model Training

-SCULLY Methodology (Contrastive Learning):
```bash
python ContrastiveLearning.py
```
-MULDER Methodology (One-vs-Rest Classification):
```bash
python OneVsRest.py
```
Both approaches output a confusion matrix and a test F1-score.

---

