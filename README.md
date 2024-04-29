# Sensitivity analysis of contact-related interventions for modeling epidemics.

## Introduction
This repository conducts sensitivity analysis on various COVID-19 models, 
exploring their behavior under different contact matrix element variations. 

## Key Investigation
- **Contact Matrix**: Analyzing the impact of contact matrix perturbation on the targets.

## targets investigated
- R0
- Epidemic size
- ICU peak size
- Hospital peak size
- Final dead size

## Methods
- **Latin Hypercube Sampling (LHS)**: Used for parameter sampling.
- **Partial Rank Correlation Coefficient (PRCC)**: Metric for sensitivity analysis.

## Usage
1. **Parameter Sampling**: Generates CSV files with parameter samples and placeholder values for target variables.
2. **Plotter**: Generates plots showing PRCC values, p-values, aggregated PRCC values and their variations.

## Folder Structure
- **data**: CSV and json files for each model (e.g., population, contact matrix, model parameters).
- **examples**: Investigated models (e.g., rost, seir, chikina, moghadas).
- **model**: Scripts for implementing the models and R0 generation.
- **prcc**: Scripts for PRCC calculation and the methods for aggregating them.
- **sampling**: Sampling scripts for contact matrix elements.
- **simulation**: Orchestrates simulation processes, calling other folders, analyzing contact manipulation.
- *target*: Output folders for models (e.g., epidemic, icu peak, final dead size, hospital peak).

## File Details
- `main.py`: Contains `Simulation` class for running sampling and PRCC calculation.
- `dataloader.py`: Loads data in an arranged format.
- `model.py`: Implements COVID-19 models.
- `r0.py`: Calculates R0 values.
- `prcc.py`: Calculates PRCC and their aggregate values.
- `sampler.py`: Conducts parameter sampling.
- `contact_manipulation.py`: Analyzes age group percentage contact manipulation on the targets.
- `plotter.py`: Generates plots based on analysis results.

## Implementation Details
This section provides insights into the implementation of the aforementioned use cases. 
Each subsection represents a specific functionality that can be executed from the `main.py` script.

## Documentation
For detailed documentation on each component of the project, refer to the respective folders and their README files.

## Requirement
This project is developed and tested with Python 3.8 or higher. Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
