from sklearn.metrics import r2_score, root_mean_squared_error, max_error, mean_absolute_error
from scipy.stats import pearsonr
import math

"""
This module contains functions to present the statistics of the regression models.
Presents the R2, RMSE, Correlation Score, Maximum Error and Mean Absolute Error.
Adaptation from TP06
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


"""
Compares two values based on the criterion. For R2, higher is better. For RMSE, Max Error and MAE, lower is better.
Its not recomended to use correlation and p-value as criteria, but they are included for completeness. 

The function returns a value between 0 and 1, where values closer to 1 indicate that val1 is better than val0 according to the criterion.
This is done by calculating the difference between the two values, normalizing it, and then applying a sigmoid function to map it to the range [0, 1].
"""
def compare_value(val0, val1, criterion="r2"):
    if criterion in ["r2", "correlation"]:
        diff = val1 - val0
    elif criterion in ["rmse", "max_error", "mae", "p_value"]:
        diff = val0 - val1
    else:
        raise ValueError("Invalid criterion: %s" % criterion)

    scale = abs(val0) + abs(val1) + 1e-12
    normalized_diff = diff / scale
    return 1.0 / (1.0 + math.exp(-8.0 * normalized_diff))
"""
Simpler version of compare_value that returns -1, 0, or 1 based on whether val1 is better than, equal to, or worse than val0 according to the criterion."""
def compare_values_simple(val0, val1, criterion="r2"):
    if compare_value(val0, val1, criterion) > 0.5:
        return 1
    elif compare_value(val0, val1, criterion) < 0.5:
        return -1
    else:
        return 0

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
This function compares the models based on a weighted combination of the criteria. 
X_test: test data features
y_test: test data labels
weights: list of weights for each criterion in a dictionary, in the order of [r2, rmse, correlation, p_value, max_error, mae]
"""

def model_weighted_score(stats, weights):
    score = 0.0
    for criterion, weight in weights.items():
        score += weight * stats[criterion]
    return score

def best_model_weighted(models, X_test, y_test, weights):
    best_model = None
    best_score = 0.0
    for model in models:
        preds = model.predict(X_test)
        model_stats = get_simple_statistics(y_test, preds)
        score = model_weighted_score(model_stats, weights)

        if score > best_score:
            best_model = model
            best_score = score

    return best_model, best_score