# sensitivity-covid19-hun

## Introduction

This repository contains code for sensitivity analysis of COVID-19 models.

The main setups investigated:
- contact matrix vs. R0 & peak size of ICU compartment
- vaccination parameters vs. final size & final size for dead compartment

For sensitivity analysis we use the following methods:
- LHS (Latin Hypercube Sampling): used for parameter sampling
- PRCC (Partial Rank Correlation Coefficient): used as a metric for sensitivity

Since model simulation (solving model equations for a time span) runs separately (CUDA code), the code has to be run 
separately for the following use cases:
1. **Parameter sampling**: generate CSV files with parameter samples along with placeholder values for target variables
2. **Analysis**: generate plots about PRCC values, demonstrations (solving the model for specified scenarios)

## Structure
For running simulation:
- `main.py`: contains class `Simulation`, which runs the sampling and PRCC calculation for different scenarios
- `dataloader.py`: contains class `DataLoader` for loading data in an arranged format

Implementation of main methods:
- `model.py`: contains model implementation
- `r0.py`: contains R0 calculation
- `prcc.py`: contains PRCC calculation

Implementation of setups for experiments:
- `sampler.py`: contains two sampling setups (contact matrix, vaccination parameters)

For analyzing results:
- `plotter.py`: contains plotting functions
- `analysis.py`: contains contact matrix manipulations for demonstrations

## Implementation details

This section contains some details about the implementation for above mentioned use cases. 
All subsections are functionalities, which can be activated from `main.py`.

### Parameter sampling
TODO

### Analysis
TODO