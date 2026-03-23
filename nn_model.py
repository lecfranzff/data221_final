# neural_network_car_price.py

import warnings
import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# -------------------------------
# SETTINGS
# -------------------------------
DATA_PATH = "cleaned_car_details_dataset.csv"
TARGET_COLUMN = "Price"

# Set to True if you want 5-fold CV results too.
# This may take longer than just the train-test evaluation.
RUN_CV = True

# Suppress convergence warnings so output is cleaner
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def preprocess_fit_transform(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fit preprocessing ONLY on training data, then apply to train and test.
    This avoids data leakage.

    Steps:
    1. Separate numeric and categorical columns
    2. Impute numeric columns with training medians
    3. Impute categorical columns with training modes
    4. One-hot encode categorical columns
    5. Align train/test columns after encoding
    6. Standardize features using training set statistics only
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # ---- Numeric imputation using training medians ----
    if numeric_cols:
        train_medians = X_train[numeric_cols].median()
        X_train[numeric_cols] = X_train[numeric_cols].fillna(train_medians)
        X_test[numeric_cols] = X_test[numeric_cols].fillna(train_medians)

    # ---- Categorical imputation using training modes ----
    if categorical_cols:
        train_modes = {}
        for col in categorical_cols:
            mode_series = X_train[col].mode(dropna=True)
            train_modes[col] = mode_series.iloc[0] if not mode_series.empty else "Unknown"

        for col in categorical_cols:
            X_train[col] = X_train[col].fillna(train_modes[col])
            X_test[col] = X_test[col].fillna(train_modes[col])

    # ---- One-hot encode AFTER split ----
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)

    # Align columns so test has exactly same columns as train
    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded,
        join="left",
        axis=1,
        fill_value=0
    )

    # ---- Standardize features using train statistics only ----
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train_encoded)
    X_test_scaled = feature_scaler.transform(X_test_encoded)

    return X_train_scaled, X_test_scaled, X_train_encoded.columns.tolist(), feature_scaler


def scale_target_fit_transform(y_train: pd.Series, y_test: pd.Series):
    """
    Scale the target for more stable neural network training.
    Fit on training target only, then apply to test target.
    """
    target_scaler = StandardScaler()

    y_train_array = np.array(y_train).reshape(-1, 1)
    y_test_array = np.array(y_test).reshape(-1, 1)

    y_train_scaled = target_scaler.fit_transform(y_train_array).ravel()
    y_test_scaled = target_scaler.transform(y_test_array).ravel()

    return y_train_scaled, y_test_scaled, target_scaler


def build_model():
    """
    Neural Network regressor for car price prediction.

    Multilayer architecture:
    - 3 hidden layers
    - ReLU activation
    - Adam optimizer
    - Early stopping for better generalization
    """
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=800,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=42
    )
    return model


def compute_metrics(y_true, y_pred):
    """
    Proposal metrics:
    - MAE
    - SSE
    """
    mae = mean_absolute_error(y_true, y_pred)
    sse = float(np.sum((np.array(y_true) - np.array(y_pred)) ** 2))
    return mae, sse


def cross_validate_neural_network(dataframe: pd.DataFrame, target_column: str, n_splits: int = 5):
    """
    Optional 5-fold cross-validation to support the success criteria
    in the proposal.
    Preprocessing is re-fit within each fold to avoid leakage.
    """
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []

    for fold_number, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_fold_train = X.iloc[train_idx].copy()
        X_fold_val = X.iloc[val_idx].copy()
        y_fold_train = y.iloc[train_idx].copy()
        y_fold_val = y.iloc[val_idx].copy()

        X_fold_train_scaled, X_fold_val_scaled, _, _ = preprocess_fit_transform(
            X_fold_train,
            X_fold_val
        )

        y_fold_train_scaled, _, target_scaler = scale_target_fit_transform(
            y_fold_train,
            y_fold_val
        )

        model = build_model()
        model.fit(X_fold_train_scaled, y_fold_train_scaled)

        pred_scaled = model.predict(X_fold_val_scaled)
        pred_original = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

        fold_mae, fold_sse = compute_metrics(y_fold_val, pred_original)

        fold_results.append({
            "Fold": fold_number,
            "MAE": fold_mae,
            "SSE": fold_sse
        })

    return pd.DataFrame(fold_results)


# -------------------------------
# MAIN PROGRAM
# -------------------------------
def main():
    # Load cleaned dataset
    df = pd.read_csv(DATA_PATH)

    # Basic validation
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the dataset.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # 70/30 split with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    # Preprocess after split (fit on train only)
    X_train_scaled, X_test_scaled, feature_names, feature_scaler = preprocess_fit_transform(
        X_train,
        X_test
    )

    # Scale target using train only
    y_train_scaled, y_test_scaled, target_scaler = scale_target_fit_transform(y_train, y_test)

    # Build and train NN model
    model = build_model()
    model.fit(X_train_scaled, y_train_scaled)

    # Predict on test set
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Evaluate using proposal metrics
    test_mae, test_sse = compute_metrics(y_test, y_pred)

    print("=" * 60)
    print("NEURAL NETWORK REGRESSION FOR USED CAR PRICE PREDICTION")
    print("=" * 60)
    print(f"Dataset shape: {df.shape}")
    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows : {X_test.shape[0]}")
    print(f"Number of engineered features after encoding: {len(feature_names)}")
    print()

    print("Test Set Evaluation")
    print("-" * 60)
    print(f"MAE: {test_mae:.4f}")
    print(f"SSE: {test_sse:.4f}")
    print()

    # Show sample predictions
    results_df = pd.DataFrame({
        "Actual Price": y_test.values,
        "Predicted Price": y_pred,
        "Absolute Error": np.abs(y_test.values - y_pred)
    })

    print("Sample Predictions (first 10 rows)")
    print("-" * 60)
    print(results_df.head(10).to_string(index=False))
    print()

    # Optional cross-validation
    if RUN_CV:
        print("Running 5-fold cross-validation...")
        cv_results = cross_validate_neural_network(df, TARGET_COLUMN, n_splits=5)

        print()
        print("Cross-Validation Results")
        print("-" * 60)
        print(cv_results.to_string(index=False))
        print()
        print("Cross-Validation Averages")
        print("-" * 60)
        print(f"Average CV MAE: {cv_results['MAE'].mean():.4f}")
        print(f"Average CV SSE: {cv_results['SSE'].mean():.4f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()