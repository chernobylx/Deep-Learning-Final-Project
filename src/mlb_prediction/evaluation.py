"""Test-set evaluation and prediction diagnostics."""

import numpy as np
from matplotlib import pyplot as plt

from .features import TARGET_COLUMNS
from .preprocessing import inverse_transform_predictions


def evaluate_model(model, X_test, y_test, scaler_dict=None):
    """Evaluate a trained model on the test set.

    Prints MAE/RMSE per target and shows predicted-vs-actual scatter plots.

    Parameters
    ----------
    model : keras.Model
        Trained model.
    X_test, y_test : pd.DataFrame
        Test features (already normalized) and targets (original scale).
    scaler_dict : dict, optional
        Scaler dictionary containing a 'targets' scaler if the targets
        were normalized during training.

    Returns
    -------
    (results, predictions) : tuple of (dict, np.ndarray)
    """
    predictions = model.predict(X_test)

    if scaler_dict and "targets" in scaler_dict:
        predictions = inverse_transform_predictions(predictions, scaler_dict, "targets")
        y_test_original = inverse_transform_predictions(y_test, scaler_dict, "targets")
    else:
        y_test_original = y_test

    results = {}
    for i, target in enumerate(TARGET_COLUMNS):
        mae = np.mean(np.abs(predictions[:, i] - y_test_original.iloc[:, i]))
        mse = np.mean((predictions[:, i] - y_test_original.iloc[:, i]) ** 2)
        results[target] = {"MAE": mae, "MSE": mse, "RMSE": np.sqrt(mse)}

    print("\nModel Evaluation Results:")
    print("=" * 50)
    for target, metrics in results.items():
        print(f"\n{target}:")
        print(f"  MAE: {metrics['MAE']:.3f}")
        print(f"  RMSE: {metrics['RMSE']:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, target in enumerate(TARGET_COLUMNS):
        actual = y_test_original.iloc[:, i]
        axes[i].scatter(actual, predictions[:, i], alpha=0.5)
        axes[i].plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--", lw=2)
        axes[i].set_xlabel(f"Actual {target}")
        axes[i].set_ylabel(f"Predicted {target}")
        axes[i].set_title(f"{target} - MAE: {results[target]['MAE']:.3f}")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results, predictions
