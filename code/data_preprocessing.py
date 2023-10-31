# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Union, Callable, Optional, Set
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from dowhy.gcm.anomaly_scorer import AnomalyScorer
from dowhy.gcm.anomaly_scorers import (
    MeanDeviationScorer,
    ITAnomalyScorer,
    MedianDeviationScorer,
    RescaledMedianCDFQuantileScorer,
)

# === BASIC PREPROCESSING === #


def map_df(df):
    df_new = df.copy(deep=True)
    df_new.index.names = ["time"]
    columns = ["_".join([c[0], c[1], c[2]]) for c in df_new.columns]
    df_new.columns = columns
    return df_new


def reduce_df(df: pd.DataFrame, metric: str, statistic: str):
    data_matrix = df.loc[:, (slice(None), [metric], [statistic])]
    # Now we can focus on the microservices as only column since metric and statistic are fixed.
    data_matrix.columns = [c[0] for c in data_matrix.columns]
    return data_matrix


def marginalize_node(graph: nx.DiGraph, node: str):
    children = graph.successors(node)
    for child in children:
        graph.add_edges_from((n, child) for n in graph.predecessors(node))
    graph.remove_node(node)


def impute_df(df: pd.DataFrame, method: str = "mean", fill: float = -1):
    """
    Wrapper around very simple imputation methods.

    Args:
        df: Pandas DataFrame in which to impute NaNs.
        method: How NaNs should be imputed. If 'mean' then each is replaced by the mean of the
            remaining values of the same microservice, metric and statistic. If 'interpolate' then
            pandas.DataFrame.interpolate(method='time',limit_direction='both') is used.
            if 'fill' then missing values will be replaced with the value `fill`.
        fill: Value with which to replace NaNs if `method = 'fill'`.
    """
    if method not in {"mean", "interpolate", "fill"}:
        ValueError(f"{method} is not a valid imputation method.")
    if method == "mean":
        df.fillna(df.mean(), inplace=True)
    elif method == "interpolate":
        df_index = df.index
        df.index = pd.to_datetime(df.index, unit="s")
        df.interpolate("time", limit_direction="both", inplace=True)
        df.interpolate("time", limit_direction="both", inplace=True)
        # reverting index back for consistency between imputation methods
        df.index = df_index
    elif method == "fill":
        df.fillna(fill, inplace=True)


# === ANOMALY DETECTION === #


@dataclass
class AnomalyDetectionConfig:
    """This class represents the configuration for an anomaly detector.

    Attributes:
        anomaly_scorer: A callable that produces an `AnomalyScorer` object.
        convert_to_p_value: A bool indicating whether to convert the scores to p-values. This only makes sense if the
                            anomaly_scorer returns scores. This does np.exp(-score).
        anomaly_score_threshold: The threshold used to determine whether a score is anomalous.
        description: A description of the configuration."""

    anomaly_scorer: Callable[[], AnomalyScorer]
    convert_to_p_value: bool
    anomaly_score_threshold: float
    description: str


DEFAULT_ANOMALY_DETECTION = AnomalyDetectionConfig(
    anomaly_scorer=MedianDeviationScorer,
    convert_to_p_value=False,
    anomaly_score_threshold=5,
    description="MADScore",
)


class MeanDeviationWithoutRescaling(AnomalyScorer):
    """The score here is simply the deviation from the mean."""

    def __init__(self, mean: Optional[float] = None):
        self._mean = mean

    def fit(self, X: np.ndarray) -> None:
        if self._mean is None:
            self._mean = np.mean(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        return abs(X.reshape(-1) - self._mean)


def train_anomaly_detectors(
    metrics,
    metric,
    statistic,
    anomaly_detection_config: AnomalyDetectionConfig,
) -> Dict[str, Tuple[AnomalyScorer, AnomalyDetectionConfig]]:
    """Trains an anomaly detector for each service defined by the anomaly detection config. The fitted models are
    stored as dictionary.

    :param training_end_time: The end time stamp for the training period.
    :param lookback_period: The lookback period for the training period.
    :param aggregation_period: The aggregation period for the training period.
    :param anomaly_detection_config: The configuration for the anomaly detection.
    :param metric: The metric to use for training.
    :param stat: The statistic to use for training.
    :param filename: The filename to store the trained anomaly detectors.
    :return: A dictionary mapping service names to the trained anomaly detectors with their config (tuple).
    """
    trained_ads = {}

    data_matrix = metrics.loc[:, (slice(None), [metric], [statistic])]
    # now we can focus on the microservices as only column since metric and statistic are fixed
    data_matrix.columns = [c[0] for c in data_matrix.columns]
    for c in data_matrix.columns:
        training_data = pd.DataFrame(data_matrix[c]).to_numpy().reshape(-1)
        training_data = training_data[~np.isnan(training_data)]
        if training_data.shape[0] < 10:
            warnings.warn(
                "After removing missing data, %s has fewer than 10 data points! Using no model instead."
                % c
            )
            continue
        elif np.std(training_data) == 0:
            warnings.warn(
                "The standard deviation of %s is 0. Using a trivial model instead." % c
            )
            scorer = MeanDeviationWithoutRescaling()
            scorer.fit(training_data)
            tmp_config = AnomalyDetectionConfig(
                MeanDeviationWithoutRescaling,
                False,
                0,
                anomaly_detection_config.description,
            )
        else:
            scorer = anomaly_detection_config.anomaly_scorer()
            scorer.fit(training_data)
            tmp_config = copy.deepcopy(anomaly_detection_config)

        trained_ads[c] = (scorer, tmp_config)

    return trained_ads


def estimate_anomaly_scores(
    data: pd.DataFrame,
    anomaly_scorer: AnomalyScorer,
    is_it_score: bool,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate anomaly scores based on the given data.

    :param data: A pandas dataframe containing the data to be scored.
    :param anomaly_scorer: An instance of AnomalyScorer to use for calculating the anomaly scores.
    :param is_it_score: A boolean indicating whether the scores returned by the anomaly scorer should be treated as IT
                        scores. This is, they would be converted to p-values based on exp(-log_prob)
    :param threshold: A float value indicating the threshold above which a score will be considered anomalous.
    :return: A tuple containing two numpy arrays, where the first entry is a binary decision whether a point is
             anomalous and the second entry is the corresponding score.
    """
    data = np.array(data.to_numpy()).reshape(-1)
    non_nan_values = data[~np.isnan(data)]
    scores = np.zeros(data.shape[0])
    if non_nan_values.shape[0] == 0:
        return np.array([False] * data.shape[0]), np.zeros(data.shape[0])

    tmp_scores = anomaly_scorer.score(non_nan_values).reshape(-1)
    if is_it_score:
        tmp_scores = 1 - np.exp(-tmp_scores)
    scores[~np.isnan(data)] = tmp_scores
    scores[np.isnan(scores)] = 0

    return scores > threshold, scores


def estimate_binary_decision_and_anomaly_scores(
    data_matrix: pd.DataFrame,
    anomaly_detectors: Dict[str, Tuple[AnomalyScorer, AnomalyDetectionConfig]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Computes the binary decision and anomaly scores.

    :param data_matrix: The data matrix to score.
    :param anomaly_detectors: The anomaly detectors.
    :return: Two dictionaries, where the first one is the binary decision and the second one the anomaly scores for
             each service.
    """
    binary_anomaly_indicators = {}
    anomaly_scores = {}

    for c in data_matrix.columns:
        if c not in anomaly_detectors or anomaly_detectors[c] is None:
            warnings.warn(
                "WARNING: No anomaly scorer found for %s! Will skip this metric and assume there "
                "are no anomalies." % c
            )
            tmp_indicator = np.array([False] * data_matrix.shape[0])
            tmp_scores = np.zeros(data_matrix.shape[0])
        else:
            tmp_indicator, tmp_scores = estimate_anomaly_scores(
                data_matrix[c],
                anomaly_detectors[c][0],
                anomaly_detectors[c][1].convert_to_p_value,
                anomaly_detectors[c][1].anomaly_score_threshold,
            )
        binary_anomaly_indicators[c] = tmp_indicator
        anomaly_scores[c] = tmp_scores

    return binary_anomaly_indicators, anomaly_scores


def get_anomalous_metrics_and_scores(
    normal_metrics: pd.DataFrame,
    abnormal_metrics: pd.DataFrame,
    target_node: str,
    target_metric: str,
    anomaly_detection_config: AnomalyDetectionConfig = DEFAULT_ANOMALY_DETECTION,
    statistic_of_interest: str = "Average",
    search_for_anomaly: bool = True,
):
    # 1. train anomaly detectors
    anomaly_detectors = train_anomaly_detectors(
        normal_metrics,
        target_metric,
        statistic_of_interest,
        anomaly_detection_config,
    )
    data_matrix = reduce_df(abnormal_metrics, target_metric, statistic_of_interest)

    # 2.a) Get binary indicator for each timestamp whether it was anomalous for a specific node.
    (
        binary_anomaly_indicators,
        anomaly_scores,
    ) = estimate_binary_decision_and_anomaly_scores(data_matrix, anomaly_detectors)

    # 2.b) Adjust trigger point. This is needed if the given trigger point does not exactly coincide with the indices of the
    # data matrix. Here, we just pick the third point in time to give the issues some time to trigger.

    row_index_to_analyze = 2
    initial_row_index = row_index_to_analyze

    if search_for_anomaly:
        for i in range(row_index_to_analyze, data_matrix.shape[0]):
            if binary_anomaly_indicators[target_node][i]:
                row_index_to_analyze = i
                break
    if row_index_to_analyze != initial_row_index:
        warnings.warn(
            "WARNING: The given trigger point %s was not anomalous. Using point %s instead, since this is the next "
            "timestamp that has been identified as anomalous."
            % (data_matrix.index[2], data_matrix.index[row_index_to_analyze])
        )

    if not binary_anomaly_indicators[target_node][row_index_to_analyze]:
        warnings.warn(
            "Target was not considered anomalous by the anomaly detector for the given trigger point!"
        )
        return {}, []

    return {
        c: anomaly_scores[c][row_index_to_analyze]
        for c in data_matrix.columns
        if binary_anomaly_indicators[c][row_index_to_analyze]
    }
