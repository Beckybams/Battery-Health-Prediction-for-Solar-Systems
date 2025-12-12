# Battery Health Prediction for Solar Systems

> **Project:** Battery-Health-Prediction-for-Solar-Systems

## Overview

This repository implements tools and example code to predict battery health (State of Health — SoH) for batteries used in solar photovoltaic (PV) systems. The project includes synthetic data generation, model training (classical ML and deep learning examples), evaluation scripts, and simple deployment/inference examples. The goal is to help operators forecast battery degradation, schedule maintenance, and extend system lifetime.

## Key Features

* Synthetic dataset generator that simulates realistic PV charge/discharge cycles and environmental effects.
* Example preprocessing and feature engineering pipelines.
* Baseline models: Linear Regression, Random Forest, Gradient Boosting.
* Neural network example using PyTorch/Keras for time-series prediction.
* Training, hyperparameter tuning, and evaluation scripts (MAE, RMSE, R²).
* Notebook demos and a minimal inference API example (Flask/FastAPI).

## Repository Structure

```
Battery-Health-Prediction-for-Solar-Systems/
├─ data/                      # sample and synthetic datasets
├─ notebooks/                 # exploratory analyses and demos
├─ src/
│  ├─ data_generation.py      # synthetic data generator
│  ├─ preprocessing.py        # cleaning & feature engineering
│  ├─ models/
│  │  ├─ baseline_models.py   # sklearn model training & utilities
│  │  └─ nn_model.py          # neural network training script
│  ├─ train.py                # unified training interface
│  ├─ evaluate.py             # evaluation metrics & plots
│  └─ inference_api.py        # simple REST inference example
├─ experiments/               # saved model checkpoints & logs
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## Getting Started

### Prerequisites

* Python 3.8+
* pip (or conda)

### Installation

```bash
git clone https://github.com/<your-org>/Battery-Health-Prediction-for-Solar-Systems.git
cd Battery-Health-Prediction-for-Solar-Systems
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Quickstart — Generate Synthetic Data

Run the generator to create a toy dataset for experimentation:

```bash
python src/data_generation.py --output data/synthetic_battery_data.csv --n-systems 200 --days 365
```

The generator simulates PV production, load profiles, ambient temperature, and battery charge/discharge cycles. It produces timestamps and per-system features plus the target `SoH`.

## Train a Baseline Model

Example training with a Random Forest baseline:

```bash
python src/train.py --model random_forest --data data/synthetic_battery_data.csv --out experiments/rf_model.pkl
```

This script will run preprocessing, fit the chosen model, save the trained model, and output evaluation metrics to `experiments/`.

## Neural Network Example

A recurrent or convolutional model may capture temporal degradation patterns better. Example (PyTorch):

```bash
python src/models/nn_model.py --data data/synthetic_battery_data.csv --epochs 50 --batch-size 64
```

## Inference API

Start the lightweight REST API for single-sample predictions (Flask or FastAPI as configured):

```bash
python src/inference_api.py --model experiments/rf_model.pkl --host 0.0.0.0 --port 8000
```

Request example (JSON):

```json
{
  "system_id": "sys_001",
  "timestamp": "2025-01-01T12:00:00",
  "features": {"soc": 0.85, "voltage": 12.3, "temp": 28.1, "daily_discharge_kwh": 3.4}
}
```

Response:

```json
{"predicted_SoH": 0.92}
```

## Evaluation Metrics

Use standard regression metrics to evaluate model performance:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Coefficient of Determination (R²)

Scripts produce plots comparing predicted vs actual SoH and residual diagnostics in `experiments/plots/`.

## Tips & Best Practices

* Real battery data (voltage curves, impedance, temperature) will outperform synthetic training — treat synthetic data as a starting point.
* Normalize features per system or use domain-adaptive scaling when mixing systems with different capacities.
* Consider time-windowed models (sliding windows) or sequence models (LSTM/Transformer) for long-term degradation forecasting.
* Use cross-validation grouped by system ID to avoid data leakage.

## Example Commands

* Generate data: `python src/data_generation.py --n-systems 500 --days 730`
* Train baseline: `python src/train.py --model xgboost --data data/synthetic_battery_data.csv`
* Evaluate: `python src/evaluate.py --model experiments/xgboost_model.pkl --data data/test_set.csv`

## Contributing

Contributions welcome! Please follow these steps:

1. Fork the repo and create a feature branch.
2. Add tests and update `requirements.txt` if needed.
3. Create a pull request with a clear description of changes.

## Roadmap (ideas)

* Add support for real-world battery datasets and import utilities.
* Implement impedance-based features and cycle-count extraction.
* Uncertainty quantification (e.g., Bayesian models, quantile regression).
* Dashboard for monitoring system fleet SoH and alerts.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Contact

For questions or collaboration, open an issue or contact the maintainer: `maintainer@example.com`.

---

*Generated README — adjust commands, paths, and API details to match your environment.*
