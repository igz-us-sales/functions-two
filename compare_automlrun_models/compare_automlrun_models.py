import time
from typing import Dict

import cloudpickle
import mlrun
from mlrun.frameworks.auto_mlrun.auto_mlrun import AutoMLRun
import numpy as np
import pandas as pd
from tensorflow import keras
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import ProbClassificationPerformanceTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import ProbClassificationPerformanceProfileSection
from evidently.pipeline.column_mapping import ColumnMapping

COMPARISON_METRICS = {
    "accuracy" : "largest",
    "precision" : "largest",
    "recall" : "largest",
    "f1" : "largest",
    "roc_auc" : "largest",
    "log_loss" : "smallest"
}

EVIDENTLY_MODEL_NAMES = ["reference", "current"]


def load_models(reference_model_path: mlrun.DataItem, current_model_path: mlrun.DataItem) -> tuple:
    """
    Load reference and current model from local/remote locations.

    :param reference_model_path: Path to reference model.
    :param current_model_path:   Path to current model.

    :return: Reference and current models.
    """
    reference_model = AutoMLRun.load_model(model_path=reference_model_path.artifact_url).model
    current_model = AutoMLRun.load_model(model_path=current_model_path.artifact_url).model

    return reference_model, current_model


def format_predictions(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    reference_preds: pd.DataFrame,
    current_preds: pd.DataFrame,
    class_mappings: Dict[int, str],
    target: str = "target",
) -> tuple:
    """
    Format data and predictions in the right format for Evidently analysis.

    :param data:            Raw input data.
    :param reference_preds: Prediction probas per class for the reference model.
    :param current_preds:   Prediction probas per class for the current model.
    :param target:          Name of target variable in dataset.
    :param class_mappings:  Mapping of integer model output to class name.

    :return: Formatted predictions for reference and current models.
    """
    # Replace numerical targets with class names
    data = X_test.copy()
    data[target] = [class_mappings[str(i)] for i in y_test.idxmax(axis=1)]

    # Format predictions per model
    reference_preds_formatted = pd.concat([data, reference_preds], axis=1)
    current_preds_formatted = pd.concat([data, current_preds], axis=1)

    return reference_preds_formatted, current_preds_formatted


def generate_html_report(
    context: mlrun.MLClientCtx,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping,
    verbose_level: int,
) -> None:
    """
    Generate HTML report with evaluation metrics for model and data.

    :param context:        MLRun context.
    :param reference_data: Predictions for reference model formatted for Evidently
    :param current_data:   Predictions for current model formatted for Evidently
    :param column_mapping: Mapping for columns and prediction target for Evidently
    :param verbose_level:  Optional parameter to include additional elements in HTML
                           report such as data quality, class distribution,
                           ROC curve, and Precision-Recall curve.
    """
    # Generate HTML report
    prob_classification_dashboard = Dashboard(
        tabs=[ProbClassificationPerformanceTab(verbose_level=verbose_level)]
    )

    prob_classification_dashboard.calculate(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    # Log HTML report
    html_path = f"prob_classification_dashboard_{int(time.time())}.html"
    prob_classification_dashboard.save(html_path)
    context.log_artifact(
        "prob_classification_dashboard",
        local_path=html_path,
    )


def log_metrics_best_model(
    context: mlrun.MLClientCtx,
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping,
    comparison_metric: str,
) -> str:
    """
    Log evaluation metrics per model and specify best model based
    on comparison metric

    :param context:           MLRun context.
    :param reference_data:    Predictions for reference model formatted for Evidently
    :param current_data:      Predictions for current model formatted for Evidently
    :param column_mapping:    Mapping for columns and prediction target for Evidently
    :param comparison_metric: Desired metric to compare models. Includes accuracy,
                              precision, recall, f1, roc_auc, and log_loss.
                              
    :return: Best model - reference or current
    """
    # Calculate evaluation metrics
    prob_classification_profile = Profile(
        sections=[ProbClassificationPerformanceProfileSection()]
    )

    prob_classification_profile.calculate(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    prob_classification_metrics = prob_classification_profile.object()[
        "probabilistic_classification_performance"
    ]["data"]["metrics"]

    # Log metrics
    for model in EVIDENTLY_MODEL_NAMES:
        for metric in COMPARISON_METRICS.keys():
            context.log_result(
                f"{model}_{metric}", prob_classification_metrics[model][metric]
            )
    
    # Return best model based on comparison metric
    best_model = get_best_model(
        reference_comparison_metric=context.results[f"reference_{comparison_metric}"],
        current_comparison_metric=context.results[f"current_{comparison_metric}"],
        comparison_direction=COMPARISON_METRICS[comparison_metric]
    )
    
    context.log_result("best_model", best_model)

def get_best_model(
    reference_comparison_metric: float,
    current_comparison_metric: float,
    comparison_direction: str
) -> str:
    """
    Determine best model based on two comparison metrics and a
    comparison direction

    :param reference_comparison_metric: Comparison metric value for reference model.
    :param current_comparison_metric:   Comparison metric value for current model.
    :param comparison_direction:        Whether the metric is desired to be smallest or largest.

    :return: Best model - reference or current
    """
    if comparison_direction == "largest":
        return "reference" if reference_comparison_metric >= current_comparison_metric else "current"
    elif comparison_direction == "smallest":
        return "reference" if reference_comparison_metric <= current_comparison_metric else "current"

def load_data(data: mlrun.DataItem):
    data_path = data.url.replace("v3io://", "/v3io")
    return np.load(data_path, allow_pickle=True)
    
def compare_models(
    context: mlrun.MLClientCtx,
    reference_model_path: mlrun.DataItem,
    current_model_path: mlrun.DataItem,
    X_test_path: mlrun.DataItem,
    y_test_path: mlrun.DataItem,
    class_mappings: Dict[int, str],
    comparison_metric: str = "accuracy",
    target: str = "target",
    verbose_level: int = 0,
) -> None:
    """
    Compare two given models via Evidently using a given dataset and output
    evaluation metrics and HTML report.

    :param context:              MLRun context.
    :param reference_model_path: Local or remote path to reference model for comparison.
    :param current_model_path:   Local or remote path to current model for comparison.
    :param data_path:            Local or remote path to data for model comparison.
    :param target:               Name of target variable in dataset.
    :param class_mappings:       Mapping of integer model output to class name.
    :param comparison_metric:    Desired metric to compare models. Includes accuracy,
                                 precision, recall, f1, roc_auc, and log_loss.
    :param verbose_level:        Optional parameter to include additional elements in HTML
                                 report such as data quality, class distribution,
                                 ROC curve, and Precision-Recall curve.
    """
    
    # Check for valid evaluation_metric
    if comparison_metric not in COMPARISON_METRICS:
        raise ValueError(
            f"Comparison metric '{comparison_metric}' not found in {COMPARISON_METRICS}"
        )

    # Read data
    X_test = X_test_path.as_df()
    y_test = y_test_path.as_df()

    # Load reference and current models
    reference_model, current_model = load_models(
        reference_model_path=reference_model_path, current_model_path=current_model_path
    )

    # Get predictions per model
    class_names = list(class_mappings.values())
    reference_preds = pd.DataFrame(
        reference_model.predict(X_test), columns=class_names
    )
    current_preds = pd.DataFrame(
        current_model.predict(X_test), columns=class_names
    )

    # Format predictions for Evidently analysis
    reference_preds_formatted, current_preds_formatted = format_predictions(
        X_test=X_test,
        y_test=y_test,
        reference_preds=reference_preds,
        current_preds=current_preds,
        class_mappings=class_mappings,
        target=target
    )

    # Generate model comparison HTML report via Evidently
    column_mapping = ColumnMapping(
        target=target, prediction=list(class_mappings.values())
    )
    generate_html_report(
        context=context,
        reference_data=reference_preds_formatted,
        current_data=current_preds_formatted,
        column_mapping=column_mapping,
        verbose_level=verbose_level,
    )

    # Log evalaution metrics and best model according to comparison metric
    best_model = log_metrics_best_model(
        context=context,
        reference_data=reference_preds_formatted,
        current_data=current_preds_formatted,
        column_mapping=column_mapping,
        comparison_metric=comparison_metric,
    )