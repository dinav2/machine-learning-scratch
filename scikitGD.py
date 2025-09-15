from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
class ScikitLinearRegression:
    def __init__(self):
        self.pipeline = None
        self.feature_names = None

    def fit(self, X, y):
        # Identify columns by type
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]),
                    categorical_features,
                ),
            ]
        )
        
        # Create pipeline
        self.pipeline = Pipeline([
            ("pre", preprocessor),
            ("lr", LinearRegression())
        ])
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
        
        # Store feature names for importance
        self.feature_names = self.pipeline.named_steps['pre'].get_feature_names_out()
        
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def get_params(self):
        return self.pipeline.get_params()

    def set_params(self, **params):
        self.pipeline.set_params(**params)
    
    def metrics(self, y_true, y_pred):
        """Calculate metrics for compatibility with custom model"""
        err = y_pred - y_true
        mse = np.mean(err**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(err))
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
    
    def get_feature_importance(self, top_n=10):
        """Get feature importance (coefficients)"""
        if self.pipeline is None:
            return None
        
        coefficients = self.pipeline.named_steps['lr'].coef_
        
        # Get feature importance
        feature_importance = list(zip(self.feature_names, abs(coefficients)))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance[:top_n]
    
    def print_feature_importance(self, top_n=10):
        """Print feature importance in a nice format"""
        importance = self.get_feature_importance(top_n)
        if importance is None:
            print("No feature importance available")
            return
        
        print(f"FEATURE IMPORTANCE (Top {top_n}):")
        for i, (feature, importance_val) in enumerate(importance):
            print(f"  {i+1:2d}. {feature}: {importance_val:.4f}")
        print()
    
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
        from data_loader import DataLoader
        
        # Set random seed
        np.random.seed(seed)
        
        # Load data
        data_loader = DataLoader()
        X_train, y_train, X_test, y_test = data_loader.load_sklearn_format()
        
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
        
        # Plot predictions
        self.plot_predictions(y_test, y_test_pred)
        
        # Print feature importance
        self.print_feature_importance()
        
        # Example prediction
        i = np.random.randint(0, len(X_test))
        x_example = X_test.iloc[i:i+1]
        y_true_example = y_test.iloc[i]
        y_pred_example = self.predict(x_example)[0]
        
        print("EXAMPLE PREDICTION:")
        print(f"  Sample {i}:")
        print(f"  Actual: ${y_true_example:,.0f}")
        print(f"  Predicted: ${y_pred_example:,.0f}")
        print(f"  Error: ${abs(y_pred_example - y_true_example):,.0f}")
        
        return train_metrics, test_metrics


if __name__ == "__main__":
    model = ScikitLinearRegression()
    train_metrics, test_metrics = model.run_full_example()


