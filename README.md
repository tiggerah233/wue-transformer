# WUE Prediction with Transformer

## Overview
This project aims to model and predict Water Use Efficiency (WUE)
using multi-source remote sensing and meteorological data.

The workflow integrates data extraction from Google Earth Engine (GEE)
with deep learning models (Transformer) for time-series prediction.

This project is part of ongoing research in hydrology and AI-driven environmental modeling.

---

## Data Source
- Remote sensing data from Google Earth Engine (GEE)
- Meteorological variables:
  - Precipitation
  - Temperature
  - Vegetation indices (e.g., NDVI)
  - Evapotranspiration

---

## Methodology
- Time-series dataset construction from multi-source data
- Data preprocessing:
  - Temporal alignment
  - Missing value handling
  - Feature engineering
- Model:
  - Transformer-based architecture for sequence modeling
- Training:
  - Supervised learning for WUE regression

---

## Project Structure
├── main.py        # training & evaluation pipeline
├── config.py      # configuration and hyperparameters
---

## Current Progress
- [x] Data collection and preprocessing pipeline (GEE)
- [x] Baseline Transformer model implementation
- [x] Initial training and validation
- [ ] Model optimization and tuning (ongoing)
- [ ] Experiment tracking and evaluation improvements

---

## Future Work
- Improve model performance with advanced architectures
- Incorporate more environmental variables
- Automate feature selection and hyperparameter tuning
- Develop API / web-based tool for real-world applications
- Explore AI-assisted research workflows (LLM-based analysis & reporting)

---

## Tech Stack
- Python
- PyTorch
- Google Earth Engine (GEE)

---

## Notes
This repository is under active development.
