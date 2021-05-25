# Generated by nuclio.export.NuclioExporter

import os
import pandas as pd
import numpy as np
import scipy as sp
import pickle
import datetime

import v3io_frames as v3f

import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer


def to_observations(context, t, u, key):
    t = (
        t.apply(lambda row: f"{'_'.join([str(row[val]) for val in t.columns])}", axis=1)
        .value_counts()
        .sort_index()
    )
    u = (
        u.apply(lambda row: f"{'_'.join([str(row[val]) for val in u.columns])}", axis=1)
        .value_counts()
        .sort_index()
    )

    joined_uniques = pd.DataFrame([t, u]).T.fillna(0).sort_index()
    joined_uniques.columns = ["t", "u"]

    t_obs = joined_uniques.loc[:, "t"]
    u_obs = joined_uniques.loc[:, "u"]

    t_pdf = t_obs / t_obs.sum()
    u_pdf = u_obs / u_obs.sum()

    context.log_dataset(f"{key}_t_pdf", pd.DataFrame(t_pdf), format="parquet")
    context.log_dataset(f"{key}_u_pdf", pd.DataFrame(u_pdf), format="parquet")
    return t_pdf, u_pdf


def tvd(t, u):
    return sum(abs(t - u)) / 2


def helinger(t, u):
    return (np.sqrt(np.sum(np.power(np.sqrt(t) - np.sqrt(u), 2)))) / np.sqrt(2)


def kl_divergence(t, u):
    t_u = np.sum(np.where(t != 0, t * np.log(t / u), 0))
    u_t = np.sum(np.where(u != 0, u * np.log(u / t), 0))
    return t_u + u_t


def all_metrics(t, u):
    return tvd(t, u), helinger(t, u), kl_divergence(t, u)


def drift_magnitude(
    context,
    t: pd.DataFrame,
    u: pd.DataFrame,
    label_col=None,
    prediction_col=None,
    discretizers: dict = None,
    n_bins=5,
    stream_name: str = "some_stream",
    results_tsdb_container: str = "bigdata",
    results_tsdb_table: str = "concept_drift/drift_magnitude",
):
    """Drift magnitude metrics
       Computes drift magnitude metrics between base dataset t and dataset u.
       Metrics:
        - TVD (Total Variation Distance)
        - Helinger
        - KL Divergence

    :param context: MLRun context
    :param t: Base dataset for the drift metrics
    :param u: Test dataset for the drift metrics
    :param label_col: Label colum in t and u
    :param prediction_col: Predictions column in t and u
    :param discritizers: Dictionary of dicsritizers for the features if available
                         (Created automatically if not provided)
    :param n_bins: Number of bins to be used for histrogram creation from continuous variables
    :param stream_name: Output stream to push metrics to
    :param results_tsdb_container: TSDB table container to push metrics to
    :param results_tsdb_table: TSDB table to push metrics to
    """

    v3io_client = v3f.Client("framesd:8081", container=results_tsdb_container)
    try:
        v3io_client.create("tsdb", results_tsdb_table, if_exists=1, rate="1/s")
    except:
        v3io_client.create(
            "tsdb", results_tsdb_table, if_exists=1, attrs={"rate": "1/s"}
        )

    df_t = t.as_df()
    df_u = u.as_df()

    drop_columns = []
    if label_col is not None:
        drop_columns.append(label_col)
    if prediction_col is not None:
        drop_columns.append(prediction_col)

    continuous_features = df_t.select_dtypes(["float"])
    if discretizers is None:
        discretizers = {}
        for feature in continuous_features.columns:
            context.logger.info(f"Fitting discretizer for {feature}")
            discretizer = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy="uniform"
            )

            discretizer.fit(continuous_features.loc[:, feature].values.reshape(-1, 1))
            discretizers[feature] = discretizer
    os.makedirs(context.artifact_path, exist_ok=True)
    discretizers_path = os.path.abspath(f"{context.artifact_path}/discritizer.pkl")
    with open(discretizers_path, "wb") as f:
        pickle.dump(discretizers, f)
    context.log_artifact("discritizers", target_path=discretizers_path)
    context.logger.info("Discretizing featuers")
    for feature, discretizer in discretizers.items():
        df_t[feature] = discretizer.transform(
            df_t.loc[:, feature].values.reshape(-1, 1)
        )
        df_u[feature] = discretizer.transform(
            df_u.loc[:, feature].values.reshape(-1, 1)
        )
        df_t[feature] = df_t[feature].astype("int")
        df_u[feature] = df_u[feature].astype("int")
    context.log_dataset("t_discrete", df_t, format="parquet")
    context.log_dataset("u_discrete", df_u, format="parquet")

    context.logger.info("Compute prior metrics")

    results = {}
    t_prior, u_prior = to_observations(
        context,
        df_t.drop(drop_columns, axis=1),
        df_u.drop(drop_columns, axis=1),
        "features",
    )
    results["prior_tvd"], results["prior_helinger"], results["prior_kld"] = all_metrics(
        t_prior, u_prior
    )

    if prediction_col is not None:
        context.logger.info("Compute prediction metrics")
        t_predictions = pd.DataFrame(df_t.loc[:, prediction_col])
        u_predictions = pd.DataFrame(df_u.loc[:, prediction_col])
        t_class, u_class = to_observations(
            context, t_predictions, u_predictions, "prediction"
        )
        (
            results["prediction_shift_tvd"],
            results["prediction_shift_helinger"],
            results["prediction_shift_kld"],
        ) = all_metrics(t_class, u_class)

    if label_col is not None:
        context.logger.info("Compute class metrics")
        t_labels = pd.DataFrame(df_t.loc[:, label_col])
        u_labels = pd.DataFrame(df_u.loc[:, label_col])
        t_class, u_class = to_observations(context, t_labels, u_labels, "class")
        (
            results["class_shift_tvd"],
            results["class_shift_helinger"],
            results["class_shift_kld"],
        ) = all_metrics(t_class, u_class)

    for key, value in results.items():
        if value == float("inf"):
            context.logger.info(f"value: {value}")
            results[key] = 10
    for key, result in results.items():
        context.log_result(key, round(result, 3))

    now = pd.to_datetime(str(datetime.datetime.now()))
    now

    results["timestamp"] = pd.to_datetime(str((datetime.datetime.now())))
    context.logger.info(f"Timestamp: {results['timestamp']}")
    results["stream"] = stream_name
    results_df = pd.DataFrame(
        data=[list(results.values())], columns=list(results.keys())
    )
    results_df = results_df.set_index(["timestamp", "stream"])
    v3io_client.write("tsdb", results_tsdb_table, dfs=results_df)
