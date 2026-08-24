"""Hyperparameter search and final model training."""

import keras_tuner as kt
from matplotlib import pyplot as plt
from tensorflow import keras

from .model import build_model


def create_bayesian_tuner(input_dim, output_dim=4, project_name="mlb_prediction_bayesian",
                          trials=32, init_points=16):
    """Create a Bayesian Optimization tuner over the model search space."""
    return kt.BayesianOptimization(
        lambda hp: build_model(hp, input_dim, output_dim),
        objective="val_loss",
        max_trials=trials,
        num_initial_points=init_points,
        directory="keras_tuner",
        project_name=project_name,
        overwrite=True,
    )


def run_hyperparameter_search(X_train, y_train, X_val, y_val,
                              trials=32, init_points=16, epochs=100, batch_size=64):
    """Run Bayesian hyperparameter search and return the best untrained model.

    Returns
    -------
    (best_model, best_hps, tuner)
    """
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    tuner = create_bayesian_tuner(input_dim, output_dim, trials=trials, init_points=init_points)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
    )

    print("Starting hyperparameter search...")
    tuner.search(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    print("\nBest Hyperparameters:")
    print(f"Number of layers: {best_hps.get('n_layers')}")
    print(f"Optimizer: {best_hps.get('optimizer')}")
    print(f"Learning rate: {best_hps.get('learning_rate')}")
    print("\nLayer:")
    print(f"  Units: {best_hps.get('units')}")
    print(f"  Activation: {best_hps.get('activation')}")
    print(f"  Batch Norm: {best_hps.get('batch_norm')}")
    print(f"  Dropout: {best_hps.get('dropout')}")

    return best_model, best_hps, tuner


def train_final_model(best_model, X_train, y_train, X_val, y_val,
                      epochs=200, batch_size=64, plot_history=True,
                      checkpoint_path="best_mlb_model.h5"):
    """Train the final model with early stopping, LR scheduling, and checkpointing."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=50, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor="val_loss", save_best_only=True
    )

    print("\nTraining final model...")
    history = best_model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1,
    )

    if plot_history:
        plot_training_history(history)

    return history


def plot_training_history(history):
    """Plot loss and MAE curves for training and validation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Training Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_title("Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["mae"], label="Training MAE")
    axes[1].plot(history.history["val_mae"], label="Validation MAE")
    axes[1].set_title("Model MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
