import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

path = kagglehub.dataset_download("camnugent/california-housing-prices")

# Dataset
DATA_PATH = path+"/housing.csv"
TARGET    = "median_house_value"
TEST_RATIO = 0.2
SEED = 42
np.random.seed(SEED)

# Hiperparametros
LR     = 0.01       # learning rate
EPOCHS = 5000       # iteraciones

df = pd.read_csv(DATA_PATH)

category_cols = df.select_dtypes(include=["object"]).columns

features = pd.get_dummies(
    df.drop(columns=['median_house_value']),
    columns=category_cols,
    drop_first=True,
    dtype=float
)

y = df['median_house_value'].astype(float)

data = pd.concat([features, y], axis=1).dropna()

# Splitting train/test

idx = np.arange(len(data))
random = np.random.default_rng(SEED)
random.shuffle(idx)

n_test = int(len(idx) * TEST_RATIO)
test_idx = idx[:n_test]
train_idx = idx[n_test:]

test = data.iloc[test_idx]
train = data.iloc[train_idx]

X_train = train[features.columns].values.astype(float)
y_train = train["median_house_value"].values.astype(float)
X_test = test[features.columns].values.astype(float)
y_test = test["median_house_value"].values.astype(float)

# Estandarizar

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_std[X_std == 0] = 1.0

X_train_std = (X_train - X_mean) / X_std
X_test_std  = (X_test  - X_mean) / X_std

# Definición de funciones

def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])

def predict_linear(X, w):
    return add_bias(X) @ w

def mse_loss(X, y, w, l2=0.0):
    y_hat = predict_linear(X, w)
    err = y_hat - y
    mse = np.mean(err**2)
    if l2 > 0.0:
        mse += l2 * np.sum(w[:-1]**2)  # no regularizar el bias
    return mse

def fit_gd(X, y, lr=0.05, epochs=3000, l2=0.5, verbose=False):
    Xb = add_bias(X)
    n, d = Xb.shape
    w = np.zeros(d)
    losses = []
    for ep in range(epochs):
        y_hat = Xb @ w
        grad = (Xb.T @ (y_hat - y)) / n
        if l2 > 0.0:
            reg = 2.0 * l2 * w
            reg[-1] = 0.0
            grad += reg
        w -= lr * grad
        if ep % max(1, epochs // 100) == 0 or ep == epochs - 1:
            losses.append(mse_loss(X, y, w, l2=l2))
            if verbose and ep % max(1, epochs // 10) == 0:
                print(f"[{ep:4d}] MSE={losses[-1]:.4f}")
    return w, np.array(losses)

def metrics(y, y_hat):
    err = y_hat - y
    mse = np.mean(err**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err))
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

# Entrenamiento del modelo

w, losses = fit_gd(X_train_std, y_train, lr=LR, epochs=EPOCHS, verbose=False)
y_train_hat_gd = predict_linear(X_train_std, w)
y_test_hat_gd  = predict_linear(X_test_std,  w)

gd_train_metrics = metrics(y_train, y_train_hat_gd)
gd_test_metrics  = metrics(y_test,  y_test_hat_gd)

print("=== GD (train)  ===", gd_train_metrics)
print("=== GD (test)   ===", gd_test_metrics)

# Plot de la pérdida

plt.figure(figsize=(6,4))
plt.plot(losses)
plt.xlabel("Checkpoints (≈1% de epochs)")
plt.ylabel("MSE (train)")
plt.title("Curva de pérdida — GD")
plt.grid(True)
plt.show()

# Plot de la predicción
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_test_hat_gd, alpha=0.6)
mx = float(max(y_test.max(), y_test_hat_gd.max()))
mn = float(min(y_test.min(), y_test_hat_gd.min()))
plt.plot([mn, mx], [mn, mx])
plt.xlabel("y_real (test)")
plt.ylabel("y_pred (GD)")
plt.title("y_pred vs y_real — GD")
plt.grid(True)
plt.show()

# Predecir ejemplo con el modelo entrenado

i = np.random.randint(0, len(X_test))
x_std = X_test_std[i:i+1]
y_true = y_test[i]

y_hat_gd  = float(predict_linear(x_std, w))

print("Ejemplo i =", i)
print("y_real:", y_true)
print("GD   y_pred:", y_hat_gd,  "| abs err:", abs(y_hat_gd - y_true))