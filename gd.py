import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import DataLoader



class CustomLinearRegression:
    def __init__(self, lr=0.01, epochs=5000, l2=0.5, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.verbose = verbose
        self.w = None
        self.losses = None
        self.X_mean = None
        self.X_std = None
        
    def add_bias(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def predict_linear(self, X, w):
        return self.add_bias(X) @ w

    def mse_loss(self, X, y, w, l2=0.0):
        y_hat = self.predict_linear(X, w)
        err = y_hat - y
        mse = np.mean(err**2)
        if l2 > 0.0:
            mse += l2 * np.sum(w[:-1]**2)  # don't regularize bias
        return mse

    def fit_gd(self, X, y, lr=0.05, epochs=3000, l2=0.5, verbose=False):
        Xb = self.add_bias(X)
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
                losses.append(self.mse_loss(X, y, w, l2=l2))
                if verbose and ep % max(1, epochs // 10) == 0:
                    print(f"[{ep:4d}] MSE={losses[-1]:.4f}")
        return w, np.array(losses)

    def metrics(self, y, y_hat):
        err = y_hat - y
        mse = np.mean(err**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(err))
        ss_res = np.sum((y - y_hat)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    def fit(self, X, y):
        # Standardize features
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        self.X_std[self.X_std == 0] = 1.0
        
        X_std = (X - self.X_mean) / self.X_std
        self.w, self.losses = self.fit_gd(X_std, y, lr=self.lr, epochs=self.epochs, l2=self.l2, verbose=self.verbose)

    def predict(self, X):
        # Standardize using training statistics
        X_std = (X - self.X_mean) / self.X_std
        return self.predict_linear(X_std, self.w)
    
    def plot_loss_curve(self, save_path=None):
        """Plot the loss curve during training"""
        if self.losses is None:
            print("No loss history available. Train the model first.")
            return
        
        plt.figure(figsize=(8, 5))
        plt.plot(self.losses)
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title(f"{self.__class__.__name__} - Training Loss Curve")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, save_path=None):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{self.__class__.__name__} - Predictions vs Actual')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def run_full_example(self, test_ratio=0.2, seed=42):
        """Run the complete example with data loading, training, and evaluation"""
        # Set random seed
        np.random.seed(seed)
        
        # Load data
        data_loader = DataLoader()
        X_train, y_train, X_test, y_test = data_loader.load_custom_format()
        
        print(f"=== {self.__class__.__name__} COMPLETE EXAMPLE ===")
        print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test data: {X_test.shape[0]} samples")
        print()
        
        # Train model
        print("Training model...")
        self.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)
        
        # Calculate metrics
        train_metrics = self.metrics(y_train, y_train_pred)
        test_metrics = self.metrics(y_test, y_test_pred)
        
        print("RESULTS:")
        print(f"  Training - MSE: {train_metrics['MSE']:.2f}, R²: {train_metrics['R2']:.4f}")
        print(f"  Test     - MSE: {test_metrics['MSE']:.2f}, R²: {test_metrics['R2']:.4f}")
        print()
        
        # Plot loss curve
        self.plot_loss_curve()
        
        # Plot predictions
        self.plot_predictions(y_test, y_test_pred)
        
        # Example prediction
        i = np.random.randint(0, len(X_test))
        x_example = X_test[i:i+1]
        y_true_example = y_test[i]
        y_pred_example = self.predict(x_example)[0]
        
        print("EXAMPLE PREDICTION:")
        print(f"  Sample {i}:")
        print(f"  Actual: ${y_true_example:,.0f}")
        print(f"  Predicted: ${y_pred_example:,.0f}")
        print(f"  Error: ${abs(y_pred_example - y_true_example):,.0f}")
        
        return train_metrics, test_metrics


if __name__ == "__main__":
    model = CustomLinearRegression(lr=0.01, epochs=5000, l2=0.5, verbose=False)
    train_metrics, test_metrics = model.run_full_example()