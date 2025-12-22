# MLflow Guide

## Installation

Install MLflow using pip:

```bash
pip install mlflow
```

## Quick Start

### Start the Tracking Server

```bash
mlflow ui --host 0.0.0.0 --port 5000 --allowed-hosts "*"
```

This starts a local server at `http://localhost:5000`.

### Log Experiments

```python
import mlflow

# Start a run
mlflow.start_run()

# Log parameters
mlflow.log_param("learning_rate", 0.01)

# Log metrics
mlflow.log_metric("accuracy", 0.95)

# Log artifacts
mlflow.log_artifact("model.pkl")

# End run
mlflow.end_run()
```

### View Results

Open your browser and navigate to `http://localhost:5000` to view experiments, runs, and metrics.

## Key Features

- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Manage model versions and transitions
- **Projects**: Package code with dependencies
- **Models**: Save and load models in multiple formats

## Documentation

For more details, visit [MLflow Documentation](https://mlflow.org/docs/latest/).
