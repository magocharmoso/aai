from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    r2_score,
    root_mean_squared_error,
    max_error,
    mean_absolute_error,
)
from scipy.stats import pearsonr
import math

# ===== SHARED METHODS ===== #
# Regression and classification

"""
Compares two values based on the criterion. For R2, higher is better. For RMSE, Max Error and MAE, lower is better.
Its not recomended to use correlation and p-value as criteria, but they are included for completeness. 

The function returns a value between 0 and 1, where values closer to 1 indicate that val1 is better than val0 according to the criterion.
This is done by calculating the difference between the two values, normalizing it, and then applying a sigmoid function to map it to the range [0, 1].
"""
def compare_value(val0, val1, criterion="r2"):
    # Metrics where larger is better
    greater_is_better = {"r2", "correlation", "precision", "recall", "f1_score", "matthews_corrcoef"}
    # Metrics where smaller is better
    smaller_is_better = {"rmse", "max_error", "mae", "p_value"}

    if criterion in greater_is_better:
        diff = val1 - val0
    elif criterion in smaller_is_better:
        diff = val0 - val1
    else:
        raise ValueError("Invalid criterion: %s" % criterion)

    scale = abs(val0) + abs(val1) + 1e-12
    normalized_diff = diff / scale
    return 1.0 / (1.0 + math.exp(-8.0 * normalized_diff))

#Simpler version of compare_value that returns -1, 0, or 1 based on whether val1 is better than, equal to, or worse than val0 according to the criterion."""
def compare_values_simple(val0, val1, criterion="r2"):
    score = compare_value(val0, val1, criterion)
    if score > 0.5:
        return 1
    elif score < 0.5:
        return -1
    else:
        return 0



"""
=== REGRESSION METRICS ONLYs ===
"""

# Default weights for weighted scoring. Keys must match the statistics keys returned
# by `get_simple_statistics`. These are reasonable starting values; adjust as needed.
DEFAULT_WEIGHTS = {
    "r2": 1.0,
    # For metrics where lower is better, use negative weights so a smaller metric
    # increases the overall score when multiplied directly.
    "rmse": -0.5,
    "correlation": 0.0,
    "p_value": 0.0,
    "max_error": -0.5,
    "mae": -0.5,
}

"""
Gets the R2, RMSE, Correlation Score, Maximum Error and Mean Absolute Error.
"""
def get_simple_statistics(truth, preds):
    r2 = r2_score(truth, preds)
    rmse = root_mean_squared_error(truth, preds)
    correlation, pval = pearsonr(truth, preds)
    max_err = max_error(truth, preds)
    mae = mean_absolute_error(truth, preds)
    return {
        "r2": r2,
        "rmse": rmse,
        "correlation": correlation,
        "p_value": pval,
        "max_error": max_err,
        "mae": mae,
    }
"""
Presents the statistics of the regression models.
R2, RMSE, Correlation Score, Maximum Error and Mean Absolute Error.
"""
def present_simple_statistics(truth, preds):
    stats = get_simple_statistics(truth, preds)
    print(
        "R2: %6.4f\nRMSE: %6.4f\nCorrelation Score: %6.4f (p-value=%e)\nMaximum Error: %6.4f\nMean Absolute Error: %6.4f\n"
        % (
            stats["r2"],
            stats["rmse"],
            stats["correlation"],
            stats["p_value"],
            stats["max_error"],
            stats["mae"],
        )
    )




def best_model_simple(models, X_test, y_test, criterion="r2"):
    best_model_name = None
    best_score = None

    for model in models:
        preds = model.predict(X_test)
        score = get_simple_statistics(y_test, preds)[criterion]

        if best_score is None or compare_value(best_score, score, criterion) > 0.5:
            best_model_name = model.__class__.__name__
            best_score = score

    return best_model_name, best_score



"""
This method compares the models based on a weighted combination of the criteria. The weights are provided in a dictionary, where the keys are the criteria and the values are the weights. The function returns the model with the highest weighted score.
models: list of models to compare
X_test: test data features
y_test: test data labels
weights: list of weights for each criterion in a dictionary, in the order of [r2, rmse, correlation, p_value, max_error, mae]
"""

def model_weighted_score(stats, weights=None):
    if stats is None or not isinstance(stats, dict):
        raise ValueError("stats must be a dictionary as returned by get_simple_statistics")

    if weights is None:
        weights = DEFAULT_WEIGHTS

    if not isinstance(weights, dict):
        raise ValueError("weights must be a dict mapping criterion->weight")

    allowed_keys = set(stats.keys())
    score = 0.0
    total_abs_weight = 0.0

    for criterion, weight in weights.items():
        if criterion not in allowed_keys:
            raise ValueError(f"Unknown criterion '{criterion}' in weights")

        try:
            val = float(stats[criterion])
        except Exception:
            raise ValueError(f"Statistic '{criterion}' is not numeric: {stats[criterion]!r}")
        score += float(weight) * val
        total_abs_weight += abs(float(weight))

    # normalize by sum of absolute weights so scores stay comparable across weightings
    if total_abs_weight > 0:
        score = score / total_abs_weight

    return score

def best_model_weighted(models, X_test, y_test, weights=None):
    if models is None:
        raise ValueError("models must be an iterable of estimators")

    if weights is None:
        weights = DEFAULT_WEIGHTS

    best_model = None
    best_score = None

    for model in models:
        try:
            preds = model.predict(X_test)
        except Exception:
            # skip models that cannot predict on the provided X_test
            continue

        try:
            model_stats = get_simple_statistics(y_test, preds)
        except Exception:
            # skip models that fail to produce stats
            continue

        try:
            score = model_weighted_score(model_stats, weights)
        except Exception:
            # malformed weights or stats: skip this model
            continue

        if best_score is None or score > best_score:
            best_model = model
            best_score = score

    return best_model, best_score




"""
=== CLASSIFIER METRICS ONLY ===
"""

DEFAULT_WEIGHTS_CLASSIFICATION = {
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0,
    "matthews_corrcoef": 1.0,
}


def get_classification_statistics(y_test, y_pred):
    return {
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "matthews_corrcoef": matthews_corrcoef(y_test, y_pred),
    }

def present_classification_statistics(y_test, y_pred):
    metrics = get_classification_statistics(y_test, y_pred)

    print('=== DECISION TREE CLASSIFIER METRICS ===\n')
    print('Confusion Matrix:')
    print(metrics["confusion_matrix"])
    print('\nClassification Report:')
    print(metrics["classification_report"])
    print('Precision:', metrics["precision"])
    print('Recall:', metrics["recall"])
    print('F1-Score:', metrics["f1_score"])
    print('Matthews Correlation Coefficient:', metrics["matthews_corrcoef"])

# VIBE CODED GOTTA CHECK IT BUT I NEEDED TO LEAVE WORK 
def best_model_classification(models, X_test, y_test, criterion="f1_score"):
    """Select the best classifier from `models` based on a classification metric.

    Uses `get_classification_statistics` to compute metrics for each model and
    returns the model with the highest value for `criterion` along with the
    corresponding score.

    Parameters
    - models: iterable of fitted estimators supporting `.predict`
    - X_test, y_test: test set
    - criterion: one of the keys returned by `get_classification_statistics`,
      e.g. 'precision', 'recall', 'f1_score', 'matthews_corrcoef'.

    Returns (best_model, best_score) or (None, None) if no model produced a score.
    """
    if models is None:
        raise ValueError("models must be an iterable of estimators")

    best_model = None
    best_score = None

    for model in models:
        try:
            preds = model.predict(X_test)
        except Exception:
            # skip models that cannot predict on the provided X_test
            continue

        try:
            stats = get_classification_statistics(y_test, preds)
        except Exception:
            # skip models that fail to produce classification stats
            continue

        if criterion not in stats:
            raise ValueError(f"Unknown classification criterion '{criterion}'")

        try:
            score = float(stats[criterion])
        except Exception:
            # if the metric isn't numeric, skip
            continue

        if best_score is None or score > best_score:
            best_model = model
            best_score = score

    return best_model, best_score

