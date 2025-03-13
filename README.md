# Hierarchical Transformers for Opioid Use Disorder (OUD) Prediction

## Overview
This repository contains code and models for predicting opioid prescription and opioid use disorder (OUD) from electronic health records (EHR) using deep learning. The study leverages both structured and unstructured data from EHRs, employing hierarchical transformer-based models to enhance prediction accuracy.
Paper link: https://doi.org/10.1016/j.ijmedinf.2022.104979 

## Features

* Hierarchical Transformer Model: Utilizes ClinicalBERT for processing unstructured clinical notes.

* Structured and Unstructured Data Integration: Combines tabular EHR data with clinical text.

* Opioid Prescription Prediction: Identifies patients likely to be prescribed opioids.

* Opioid Use Disorder (OUD) Prediction: Detects patients at risk of developing OUD.

* Performance Metrics:

  * Opioid Prescription Prediction: F1-score = 0.88, AUC-ROC = 0.93

  * OUD Prediction: F1-score = 0.82, AUC-ROC = 0.94

## Data Source

The models are trained and evaluated on the MIMIC-III database, which contains de-identified ICU admission records, including structured patient data and clinical notes.

## Model Architecture

The predictive models consist of four key components:

1. **Structured Static Component**: Encodes static patient demographics and admission data.
2. **Structured Time-Varying Component**: Processes time-sequenced clinical events.
3. **Unstructured Clinical Notes Component**: Utilizes ClinicalBERT for feature extraction from textual data.
4. **Aggregation Component**: Merges embeddings from all components for final prediction.

## Files

*Hierarchical_Model.py* contains the pytorch code for the model with training and evaluation. 

*Data_Processing_and_Evaluation.py* contains the preprocessing steps used for the Mimic-III dataset.



