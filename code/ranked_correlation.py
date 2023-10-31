# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Callable

import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from dowhy.gcm.anomaly_scorer import AnomalyScorer
from dowhy.gcm.anomaly_scorers import (
    MeanDeviationScorer,
    ITAnomalyScorer,
    MedianDeviationScorer,
    RescaledMedianCDFQuantileScorer,
)

from rca_task import PotentialRootCause
from data_preprocessing import (
    AnomalyDetectionConfig,
    MeanDeviationWithoutRescaling,
    train_anomaly_detectors,
    estimate_anomaly_scores,
    estimate_binary_decision_and_anomaly_scores,
    get_anomalous_metrics_and_scores,
    DEFAULT_ANOMALY_DETECTION,
    reduce_df,
    impute_df,
)


def make_ranked_correlation(
    method: Callable = pearsonr,
    data_pool: str = "abnormal",
    rank_by: str = "pvalue",
    root_cause_top_k: int = 3,
    anomaly_detection_config: AnomalyDetectionConfig = DEFAULT_ANOMALY_DETECTION,
    filter_for_anomalous_metrics: bool = True,
    imputation_method: str = "drop",
):
    """
    Simple RCA method whereby the root causes are assumed to be those (anomalous) metrics which most correlate with the
        metric which is in violation with SLO.

    Args:
        method: Which correlation measure to use. Default: scipy.stats.pearsonr
        data_pool: Which data the correlations should be measured in. 'normal' if in the normal regime, 'abnormal' if
            in the abnormal regime and 'pooled' if correlations should be measured across both together. Default: 'abnormal'
        rank_by: Whether to rank by the absolute correlation measure (`rank_by = 'correlation'`), by the associated p-value
            (`rank_by = 'pvalue'`) of the measure (null hypothesis measure = 0), or by the combination of the two
            (`rank_by = 'combination') according to the score = -np.log(pvalue)*np.abs(correlation). Default: pvalue
        root_cause_top_k: The maximum number of root causes in the results. Default: 3
        anomaly_detection_config: Configuration for anomaly detection.
        filter_for_anomalous_metrics: Whether to only consider metrics detected as anomalous as potential root causes.
        imputation_method: Method for imputation. If 'drop' then NaNs are simply ignored and correlations computed without
            them. If 'mean' then each is replaced by the mean of the remaining values of the same microservice, metric and
            statistic. If 'interpolate' then pandas.DataFrame.interpolate(method='time',limit_direction='both') is used.
            Default: 'drop'
    """

    def analyze_root_causes(
        graph: nx.DiGraph,
        target_node: str,
        target_metric: str,
        target_statistic: str,
        normal_metrics: pd.DataFrame,
        abnormal_metrics: pd.DataFrame,
    ) -> List[PotentialRootCause]:
        """Method to identify potential root causes of the performance issue in target_node.

        Args:
            graph: Call graph of microservice architecture.
            target_node: Node whose SLO violoation to investigate.
            target_metric: Metric that is in violation with SLO.
            target_statistic: Statistic such as Average of the target_metric that is in violation with SLO.
            normal_metrics: Metrics of all microservices during previous normal operations.
            abnormal_metrics: Metrics of all microservices during SLO violation.

        Returns: List of potential root causes identifying nodes and assigning them scores.
        """
        statistic_of_interest = target_statistic  # We can consider different choices here in the future.
        normal_metrics = normal_metrics.copy()
        abnormal_metrics = abnormal_metrics.copy()

        if filter_for_anomalous_metrics:
            anomalous_metrics_and_scores = get_anomalous_metrics_and_scores(
                normal_metrics,
                abnormal_metrics,
                target_node,
                target_metric,
                anomaly_detection_config,
                statistic_of_interest,
            )
            if anomalous_metrics_and_scores == ({}, []):
                return []
            anomalous_metrics = list(anomalous_metrics_and_scores.keys())

        normal_metrics = reduce_df(normal_metrics, target_metric, statistic_of_interest)
        abnormal_metrics = reduce_df(
            abnormal_metrics, target_metric, statistic_of_interest
        )

        if rank_by == "correlation":
            new_method = lambda x, y: method(x, y)[0]
        elif rank_by == "pvalue":
            new_method = lambda x, y: -1 * method(x, y)[1]
        elif rank_by == "combination":

            def score_combination(x, y):
                temp = method(x, y)
                return -np.log(temp[1]) * np.abs(temp[0])

            new_method = score_combination

        if imputation_method == "drop":

            def nan_method(x, y):
                nans = np.logical_or(np.isnan(x), np.isnan(y))
                x, y = x[~nans], y[~nans]
                if len(x) < 2:
                    return np.nan
                return new_method(x, y)

            newer_method = nan_method
        else:
            newer_method = new_method
            impute_df(normal_metrics, imputation_method)
            impute_df(abnormal_metrics, imputation_method)

        if data_pool == "normal":
            data = normal_metrics

        if data_pool == "abnormal":
            data = abnormal_metrics

        if data_pool == "pooled":
            data = pd.concat([normal_metrics, abnormal_metrics])

        corrs = data.corr(method=newer_method).loc[:, target_node].drop(target_node)

        if filter_for_anomalous_metrics:
            corrs = corrs.loc[np.isin(corrs.index, anomalous_metrics)]

        corrs.dropna(inplace=True)
        corrs.sort_values(inplace=True, ascending=False)
        potential_root_causes = [
            PotentialRootCause(node, target_metric, corrs[node]) for node in corrs.index
        ]
        return potential_root_causes[:root_cause_top_k]

    return analyze_root_causes
