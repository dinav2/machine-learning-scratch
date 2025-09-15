import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from gd import CustomLinearRegression
from scikitGD import ScikitLinearRegression

def main():
    print("=== LINEAR REGRESSION MODELS COMPARISON ===\n")
    
    # Configuration
    TARGET = "median_house_value"
    TEST_RATIO = 0.2
    SEED = 42
    np.random.seed(SEED)
    
    # Load data
    print("Loading data...")
    data_loader = DataLoader()
    
    # Data for custom model (with manual scaling)
    X_train_custom, y_train_custom, X_test_custom, y_test_custom = data_loader.load_custom_format()
    
    # Scale data for custom model
    X_mean = X_train_custom.mean(axis=0)
    X_std = X_train_custom.std(axis=0)
    X_std[X_std == 0] = 1.0
    
    X_train_std = (X_train_custom - X_mean) / X_std
    X_test_std = (X_test_custom - X_mean) / X_std
    
    # Data for scikit-learn model (no scaling, done internally)
    X_train_sklearn, y_train_sklearn, X_test_sklearn, y_test_sklearn = data_loader.load_sklearn_format()
    
    print(f"Data loaded:")
    print(f"  - Training: {X_train_custom.shape[0]} samples, {X_train_custom.shape[1]} features")
    print(f"  - Test: {X_test_custom.shape[0]} samples")
    print()
    
    # === CUSTOM MODEL (GRADIENT DESCENT) ===
    print("=" * 60)
    print("TRAINING CUSTOM MODEL (GRADIENT DESCENT)")
    print("=" * 60)
    
    custom_model = CustomLinearRegression(lr=0.01, epochs=5000, l2=0.5, verbose=False)
    custom_model.fit(X_train_std, y_train_custom)
    
    # Custom predictions
    y_train_pred_custom = custom_model.predict(X_train_std)
    y_test_pred_custom = custom_model.predict(X_test_std)
    
    # Custom metrics
    custom_train_metrics = custom_model.metrics(y_train_custom, y_train_pred_custom)
    custom_test_metrics = custom_model.metrics(y_test_custom, y_test_pred_custom)
    
    print("CUSTOM MODEL RESULTS:")
    print(f"  Training - MSE: {custom_train_metrics['MSE']:.2f}, R²: {custom_train_metrics['R2']:.4f}")
    print(f"  Test     - MSE: {custom_test_metrics['MSE']:.2f}, R²: {custom_test_metrics['R2']:.4f}")
    print()
    
    # === SCIKIT-LEARN MODEL ===
    print("=" * 60)
    print("TRAINING SCIKIT-LEARN MODEL")
    print("=" * 60)
    
    sklearn_model = ScikitLinearRegression()
    sklearn_model.fit(X_train_sklearn, y_train_sklearn)
    
    # Scikit-learn predictions
    y_train_pred_sklearn = sklearn_model.predict(X_train_sklearn)
    y_test_pred_sklearn = sklearn_model.predict(X_test_sklearn)
    
    # Scikit-learn metrics (using class method)
    sklearn_train_metrics = sklearn_model.metrics(y_train_sklearn, y_train_pred_sklearn)
    sklearn_test_metrics = sklearn_model.metrics(y_test_sklearn, y_test_pred_sklearn)
    
    print("SCIKIT-LEARN MODEL RESULTS:")
    print(f"  Training - MSE: {sklearn_train_metrics['MSE']:.2f}, R²: {sklearn_train_metrics['R2']:.4f}")
    print(f"  Test     - MSE: {sklearn_test_metrics['MSE']:.2f}, R²: {sklearn_test_metrics['R2']:.4f}")
    print()
    
    # === COMPARACIÓN ===
    print("=" * 60)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 60)
    
    print(f"{'Métrica':<20} {'Custom GD':<15} {'Scikit-learn':<15} {'Diferencia':<15}")
    print("-" * 65)
    
    # Comparar métricas de prueba
    mse_diff = custom_test_metrics['MSE'] - sklearn_test_metrics['MSE']
    rmse_diff = custom_test_metrics['RMSE'] - sklearn_test_metrics['RMSE']
    mae_diff = custom_test_metrics['MAE'] - sklearn_test_metrics['MAE']
    r2_diff = custom_test_metrics['R2'] - sklearn_test_metrics['R2']
    
    print(f"{'Test MSE':<20} {custom_test_metrics['MSE']:<15.2f} {sklearn_test_metrics['MSE']:<15.2f} {mse_diff:<15.2f}")
    print(f"{'Test RMSE':<20} {custom_test_metrics['RMSE']:<15.2f} {sklearn_test_metrics['RMSE']:<15.2f} {rmse_diff:<15.2f}")
    print(f"{'Test MAE':<20} {custom_test_metrics['MAE']:<15.2f} {sklearn_test_metrics['MAE']:<15.2f} {mae_diff:<15.2f}")
    print(f"{'Test R²':<20} {custom_test_metrics['R2']:<15.4f} {sklearn_test_metrics['R2']:<15.4f} {r2_diff:<15.4f}")
    print()
    
    # Overfitting analysis
    custom_overfitting = abs(custom_train_metrics['R2'] - custom_test_metrics['R2'])
    sklearn_overfitting = abs(sklearn_train_metrics['R2'] - sklearn_test_metrics['R2'])
    
    print("OVERFITTING ANALYSIS:")
    print(f"  Custom GD overfitting gap: {custom_overfitting:.4f}")
    print(f"  Scikit-learn overfitting gap: {sklearn_overfitting:.4f}")
    print()
    
    # Determine best model
    if custom_test_metrics['R2'] > sklearn_test_metrics['R2']:
        print("BEST MODEL: Custom Gradient Descent")
        print(f"   R2 advantage: {r2_diff:.4f}")
    elif sklearn_test_metrics['R2'] > custom_test_metrics['R2']:
        print("BEST MODEL: Scikit-learn Linear Regression")
        print(f"   R2 advantage: {-r2_diff:.4f}")
    print()
    
    # === VISUALIZATIONS ===
    print("Generating visualizations...")
    
    # 1. Custom model loss curve
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(custom_model.losses)
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Loss Curve - Custom GD")
    plt.grid(True)
    
    # 2. Predictions vs Actual Values - Custom
    plt.subplot(1, 3, 2)
    plt.scatter(y_test_custom, y_test_pred_custom, alpha=0.6, label='Custom GD')
    min_val = min(y_test_custom.min(), y_test_pred_custom.min())
    max_val = max(y_test_custom.max(), y_test_pred_custom.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel("Actual Values")
    plt.ylabel("Predictions")
    plt.title("Custom GD - Predictions vs Actual")
    plt.legend()
    plt.grid(True)
    
    # 3. Predictions vs Actual Values - Scikit-learn
    plt.subplot(1, 3, 3)
    plt.scatter(y_test_sklearn, y_test_pred_sklearn, alpha=0.6, label='Scikit-learn', color='orange')
    min_val = min(y_test_sklearn.min(), y_test_pred_sklearn.min())
    max_val = max(y_test_sklearn.max(), y_test_pred_sklearn.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel("Actual Values")
    plt.ylabel("Predictions")
    plt.title("Scikit-learn - Predictions vs Actual")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Example prediction
    print("=" * 60)
    print("EXAMPLE PREDICTION")
    print("=" * 60)
    
    i = np.random.randint(0, len(X_test_custom))
    
    # Custom prediction
    x_std_example = X_test_std[i:i+1]
    y_true = y_test_custom[i]
    y_pred_custom = custom_model.predict(x_std_example)[0]
    
    # Scikit-learn prediction (we need to convert the example to sklearn format)
    # For simplicity, we use the same index but with different data
    x_sklearn_example = X_test_sklearn.iloc[i:i+1]
    y_pred_sklearn = sklearn_model.predict(x_sklearn_example)[0]
    
    print(f"Example {i}:")
    print(f"  Actual value: ${y_true:,.0f}")
    print(f"  Custom GD prediction: ${y_pred_custom:,.0f} (error: ${abs(y_pred_custom - y_true):,.0f})")
    print(f"  Scikit-learn prediction: ${y_pred_sklearn:,.0f} (error: ${abs(y_pred_sklearn - y_true):,.0f})")
    
    # Show feature importance for sklearn model
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (SCIKIT-LEARN)")
    print("=" * 60)
    sklearn_model.print_feature_importance()
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
