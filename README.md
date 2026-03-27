# Lightweight Transformer MJO

This repository contains the source code and the sample dataset for the manuscript titled **"A Lightweight Transformer Architecture with Self-Attention Mechanism for Daily Flood Hazard Classification Driven by Madden-Julian Oscillation Dynamics"**, submitted to *Computers & Geosciences*.

## Purpose
The provided code demonstrates the implementation of a lightweight Transformer architecture designed to classify daily flood hazards. The model processes hydrometeorological parameters and Madden-Julian Oscillation (MJO) dynamics to predict extreme weather anomalies in maritime tropical regions.

## Repository Contents
- `transformer_model.py`: The main Python script containing the Transformer model architecture, data preprocessing, training, and evaluation pipelines.
- `Sumbawa_Daily_2016_2025.csv`: A sample observation dataset used to perform a quick-test of the model.

## System Requirements
To execute the script, ensure you have Python 3.8 or higher installed along with the following primary libraries:
- `tensorflow` (TensorFlow/Keras)
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

## Quick-Test Instructions
To verify the functionality of the model using the provided sample dataset, please follow these steps:

1. Clone this repository to your local environment:
   ```bash
   git clone [https://github.com/romiaprianto/Lightweight-Transformer-MJO.git](https://github.com/romiaprianto/Lightweight-Transformer-MJO.git)

2. Navigate into the cloned directory:
   `cd Lightweight-Transformer-MJO`

4. Execute the main Python script:
   `python transformer_model.py`

6. The script will automatically load `Sumbawa_Daily_2016_2025.csv`, process the thermodynamic and MJO features, train the lightweight Transformer model, and output the classification metrics to the console.

## Developer
Romi Aprianto
