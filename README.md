# Linear Regression Models Comparison

A comprehensive comparison between custom Gradient Descent implementation and Scikit-learn's Linear Regression on the California Housing Prices dataset.

## Project Overview

This project implements and compares two different approaches to linear regression:

1. **Custom Gradient Descent** - A from-scratch implementation using batch gradient descent with L2 regularization
2. **Scikit-learn Linear Regression** - Using sklearn's optimized implementation with advanced preprocessing

Both models are trained on the same dataset but use different preprocessing pipelines, allowing for a fair comparison of the underlying algorithms.

## Project Structure

```
machine-learning-scratch/
├── main.py                # Main comparison script
├── gd.py                  # Custom Gradient Descent implementation
├── scikitGD.py            # Scikit-learn Linear Regression wrapper
├── data_loader.py         # Data loading and preprocessing utilities
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dinav2/machine-learning-scratch.git
   cd machine-learning-scratch
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Comparison

**Run the complete comparison:**
```bash
python main.py
```

This will:
- Load the California Housing Prices dataset
- Train both models with their respective preprocessing
- Display detailed metrics comparison
- Generate visualization plots
- Show feature importance analysis


## Individual Model Testing

### Data Loading Options

```python
from data_loader import DataLoader

# Custom format (for Gradient Descent)
X_train, y_train, X_test, y_test = data_loader.load_custom_format()

# Scikit-learn format (for sklearn models)
X_train, y_train, X_test, y_test = data_loader.load_sklearn_format()
```

## Understanding the Results

### Why Different Preprocessing?

The models use different preprocessing approaches:

- **Custom GD**: Manual scaling for numerical features and `pd.get_dummies()` for categorical features
- **Scikit-learn**: Advanced pipeline with `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder`

This difference is intentional to show how preprocessing affects model performance.