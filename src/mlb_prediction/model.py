"""Model architecture and the consistency-aware loss function."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def consistency_aware_loss(y_true, y_pred):
    """MSE loss with penalties that enforce relationships between outputs.

    The four outputs are (home, away, total, margin). Beyond a weighted
    per-output MSE, two consistency penalties push the model towards
    predictions where ``total = home + away`` and ``margin = |home - away|``.
    """
    home_true = y_true[:, 0]
    away_true = y_true[:, 1]
    total_true = y_true[:, 2]
    margin_true = y_true[:, 3]

    home_pred = y_pred[:, 0]
    away_pred = y_pred[:, 1]
    total_pred = y_pred[:, 2]
    margin_pred = y_pred[:, 3]

    home_mse = tf.reduce_mean(tf.square(home_true - home_pred))
    away_mse = tf.reduce_mean(tf.square(away_true - away_pred))
    total_mse = tf.reduce_mean(tf.square(total_true - total_pred))
    margin_mse = tf.reduce_mean(tf.square(margin_true - margin_pred))

    output_weights = {
        "home": 1.0,
        "away": 1.0,
        "total": 1.0,
        "margin": 1.0,
    }

    weighted_mse = (
        output_weights["home"] * home_mse
        + output_weights["away"] * away_mse
        + output_weights["total"] * total_mse
        + output_weights["margin"] * margin_mse
    )

    total_consistency = tf.reduce_mean(tf.square((home_pred + away_pred) - total_pred))
    margin_consistency = tf.reduce_mean(
        tf.square(tf.abs(home_pred - away_pred) - margin_pred)
    )

    consistency_weight = 0.2
    return weighted_mse + consistency_weight * (total_consistency + margin_consistency)


def build_model(hp, input_dim, output_dim=4):
    """Build a tunable Keras Sequential model for Keras Tuner.

    Parameters
    ----------
    hp : keras_tuner.HyperParameters
        Hyperparameter search space handle.
    input_dim : int
        Number of input features.
    output_dim : int
        Number of regression outputs (default: 4).

    Returns
    -------
    keras.Model
        Compiled model using :func:`consistency_aware_loss`.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    n_layers = hp.Int("n_layers", min_value=2, max_value=32, sampling="log")
    units = hp.Int("units", min_value=32, max_value=256, sampling="log")
    activation = hp.Choice("activation", values=["relu", "tanh", "elu", "prelu"])
    batch_norm = hp.Boolean("batch_norm")
    dropout_rate = hp.Float("dropout", min_value=0.0, max_value=0.5, sampling="linear")

    for _ in range(n_layers):
        model.add(layers.Dense(units))

        if activation == "prelu":
            model.add(layers.PReLU(alpha_initializer="Zeros"))
        else:
            model.add(layers.Activation(activation))

        if batch_norm:
            model.add(layers.BatchNormalization())

        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(output_dim, activation="linear"))

    optimizer_choice = hp.Choice("optimizer", values=["adam", "adamax", "nadam"])
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")

    if optimizer_choice == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == "adamax":
        optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=consistency_aware_loss, metrics=["mae", "mse"])

    return model
