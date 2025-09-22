import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_loader import DataLoader
from gd import CustomLinearRegression


def train_val_test_split(val_size: float = 0.2, test_size: float = 0.2, seed: int = 42):
    data = DataLoader(test_ratio=test_size, seed=seed)

    # Load numpy-ready data (already one-hot encoded)
    X_train_full, y_train_full, X_test, y_test = data.load_custom_format()

    # Train/validation split from training portion
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=seed
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_model(model: CustomLinearRegression, X_tr, y_tr, X_va, y_va, X_te, y_te):
    y_tr_pred = model.predict(X_tr)
    y_va_pred = model.predict(X_va)
    y_te_pred = model.predict(X_te)

    tr = model.metrics(y_tr, y_tr_pred)
    va = model.metrics(y_va, y_va_pred)
    te = model.metrics(y_te, y_te_pred)
    return tr, va, te

np.random.seed(42)

# Data splits
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split()

# Baseline model (no regularization)
baseline = CustomLinearRegression(lr=0.01, epochs=4000, l2=0.0, verbose=False)
baseline.fit(X_train, y_train)
tr_b, va_b, te_b = evaluate_model(baseline, X_train, y_train, X_val, y_val, X_test, y_test)

# Regularization sweep
l2_grid = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
val_results = []
for l2 in l2_grid:
    model = CustomLinearRegression(lr=0.01, epochs=4000, l2=l2, verbose=False)
    model.fit(X_train, y_train)
    tr_m, va_m, te_m = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)
    val_results.append((l2, tr_m, va_m, te_m, model))

# Pick best by validation MSE
best_l2, tr_best, va_best, te_best, best_model = min(
    val_results, key=lambda t: t[2]["MSE"]
)

# Print summary
def fmt(x, nd=3):
    return f"{x:.{nd}f}"

print("\n=== Best by validation MSE ===")
print(f"Best L2: {best_l2}")
print(
    f"Train: MSE={fmt(tr_best['MSE'])}, RMSE={fmt(tr_best['RMSE'])}, "
    f"MAE={fmt(tr_best['MAE'])}, R2={tr_best['R2']:.4f}"
)
print(
    f"Val  : MSE={fmt(va_best['MSE'])}, RMSE={fmt(va_best['RMSE'])}, "
    f"MAE={fmt(va_best['MAE'])}, R2={va_best['R2']:.4f}"
)
print(
    f"Test : MSE={fmt(te_best['MSE'])}, RMSE={fmt(te_best['RMSE'])}, "
    f"MAE={fmt(te_best['MAE'])}, R2={te_best['R2']:.4f}"
)



# Best model loss
plt.figure(figsize=(7, 4))
plt.plot(best_model.losses)
plt.xlabel("Epoch checkpoints")
plt.ylabel("MSE")
plt.title(f"Curva de pérdida - Mejor L2={best_l2}")
plt.grid(True)
plt.close()

# Metrics vs L2 on validation
l2_arr = np.array([t[0] for t in val_results], dtype=float)
mse_arr = np.array([t[2]["MSE"] for t in val_results], dtype=float)
rmse_arr = np.array([t[2]["RMSE"] for t in val_results], dtype=float)
mae_arr = np.array([t[2]["MAE"] for t in val_results], dtype=float)
r2_arr = np.array([t[2]["R2"] for t in val_results], dtype=float)

plt.figure(figsize=(7, 4))
plt.plot(l2_arr, mse_arr, marker="o", label="MSE")
plt.plot(l2_arr, rmse_arr, marker="o", label="RMSE")
plt.plot(l2_arr, mae_arr, marker="o", label="MAE")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("L2")
plt.ylabel("Valor de métrica")
plt.title("Validación: métricas vs L2 (log-log)")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(l2_arr, r2_arr, marker="o", label="R2")
plt.xscale("log")
plt.xlabel("L2")
plt.ylabel("R2")
plt.title("Validación: R2 vs L2 (log-x)")
plt.grid(True, which="both", ls=":")
plt.close()