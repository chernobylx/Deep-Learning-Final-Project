"""Train/validation/test splitting and feature normalization."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .features import TARGET_COLUMNS

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def create_feature_target_split(modeling_df, include_teams=False, target="all"):
    """Split the modeling dataframe into a feature matrix and target matrix.

    Parameters
    ----------
    modeling_df : pd.DataFrame
        Output of the feature-engineering pipeline.
    include_teams : bool
        Whether team-name columns are present (and must be excluded).
    target : str or list
        'all', 'scores', 'total', 'margin', or an explicit list of targets.

    Returns
    -------
    (X, y) : tuple of pd.DataFrame
    """
    if target == "all":
        target_columns = TARGET_COLUMNS
    elif target == "scores":
        target_columns = ["home_score", "away_score"]
    elif target == "total":
        target_columns = ["total"]
    elif target == "margin":
        target_columns = ["margin"]
    elif isinstance(target, list):
        target_columns = target
    else:
        raise ValueError(
            f"Invalid target: {target}. Use 'all', 'scores', 'total', 'margin', or a list."
        )

    # Exclude every target column from the features, not just the selected ones
    exclude_from_features = list(TARGET_COLUMNS)
    if include_teams:
        exclude_from_features += ["home", "away"]

    feature_columns = [col for col in modeling_df.columns if col not in exclude_from_features]
    return modeling_df[feature_columns], modeling_df[target_columns]


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """Random 70/15/15 split into train, validation, and test sets.

    Note: this is a random split, matching the notebook. For a stricter
    real-world evaluation, a chronological split should be used instead.
    """
    holdout = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=holdout, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / holdout, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_train_test_split(X_train, X_test, y_train=None, y_test=None,
                               method="standard", normalize_targets=False):
    """Normalize train and test sets, fitting scalers on training data only.

    Returns
    -------
    ``(X_train_norm, X_test_norm, scaler_dict)`` or, when
    ``normalize_targets`` is set,
    ``(X_train_norm, X_test_norm, y_train_norm, y_test_norm, scaler_dict)``.
    """
    if method not in SCALERS:
        raise ValueError(f"Unknown method: {method}. Use one of {sorted(SCALERS)}.")
    scaler_class = SCALERS[method]

    scaler_dict = {}

    feature_scaler = scaler_class()
    X_train_norm = pd.DataFrame(
        feature_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_norm = pd.DataFrame(
        feature_scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    scaler_dict["features"] = feature_scaler

    if normalize_targets and y_train is not None:
        target_scaler = scaler_class()
        y_train_norm = pd.DataFrame(
            target_scaler.fit_transform(y_train), columns=y_train.columns, index=y_train.index
        )
        y_test_norm = (
            pd.DataFrame(
                target_scaler.transform(y_test), columns=y_test.columns, index=y_test.index
            )
            if y_test is not None
            else None
        )
        scaler_dict["targets"] = target_scaler
        return X_train_norm, X_test_norm, y_train_norm, y_test_norm, scaler_dict

    return X_train_norm, X_test_norm, scaler_dict


def inverse_transform_predictions(predictions, scaler_dict, prediction_type="targets"):
    """Convert normalized predictions back to the original scale."""
    if prediction_type not in scaler_dict:
        raise ValueError(f"No scaler found for {prediction_type}")

    scaler = scaler_dict[prediction_type]

    if isinstance(predictions, pd.DataFrame):
        return pd.DataFrame(
            scaler.inverse_transform(predictions),
            columns=predictions.columns,
            index=predictions.index,
        )
    return scaler.inverse_transform(predictions)
