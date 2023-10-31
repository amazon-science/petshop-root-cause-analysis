# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from enum import Enum
from typing import Dict, List, Any, Tuple, Union, Callable, Optional, Set
import os
import warnings

from pyrca.analyzers.random_walk import RandomWalk, RandomWalkConfig

import networkx as nx
import numpy as np
import pandas as pd

from rca_task import PotentialRootCause
from dowhy.gcm.anomaly_scorer import AnomalyScorer
from dowhy.gcm.anomaly_scorers import (
    MeanDeviationScorer,
    ITAnomalyScorer,
    MedianDeviationScorer,
    RescaledMedianCDFQuantileScorer,
)
from data_preprocessing import (
    AnomalyDetectionConfig,
    MeanDeviationWithoutRescaling,
    train_anomaly_detectors,
    estimate_anomaly_scores,
    estimate_binary_decision_and_anomaly_scores,
    get_anomalous_metrics_and_scores,
    DEFAULT_ANOMALY_DETECTION,
    reduce_df,
    marginalize_node,
    impute_df,
)


def make_random_walk(
    anomaly_detection_config: AnomalyDetectionConfig = DEFAULT_ANOMALY_DETECTION,
    imputation_method: str = "mean",
    use_partial_corr: bool = False,
    rho: float = 0.1,
    num_steps: int = 10,
    num_repeats: int = 1000,
    root_cause_top_k: int = 3,
    search_for_anomaly: bool = True,
):
    """
    Wrapper around the 'random walk' RCA method as implemented in https://github.com/salesforce/PyRCA.

    Paper: https://doi.org/10.1109/CCGRID.2018.00076, https://doi.org/10.1145/3442381.3449905

    Args:
        use_partial_corr: Whether to use partial correlation when computing edge weights.
            Deafult: False
        rho: The weight from a "cause" node to a "result" node. Default: 0.1
        num_steps: The number of random walk steps in each run. Default: 10
        num_repeats: The number of random walk runs. Default: 1000
        root_cause_top_k: The maximum number of root causes in the results. Default: 3
        search_for_anomaly: A boolean indicating whether to search for anomalies in the causal graph. Default: True
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

        causal_graph = graph.reverse()

        all_anomalous_nodes_with_score = get_anomalous_metrics_and_scores(
            normal_metrics,
            abnormal_metrics,
            target_node,
            target_metric,
            anomaly_detection_config,
        )
        if all_anomalous_nodes_with_score == ({}, []):
            return []
        anomalous_nodes = list(all_anomalous_nodes_with_score.keys())

        abnormal_metrics = reduce_df(
            abnormal_metrics, target_metric, statistic_of_interest
        )

        abnormal_metrics = abnormal_metrics.loc[
            :, abnormal_metrics.columns[~abnormal_metrics.isna().all()]
        ]
        impute_df(abnormal_metrics, imputation_method)

        missing_nodes = abnormal_metrics.columns.symmetric_difference(
            causal_graph.nodes()
        )
        for node in missing_nodes:
            marginalize_node(causal_graph, node)

        causal_graph = nx.to_pandas_adjacency(causal_graph)

        model = RandomWalk(
            config=RandomWalkConfig(
                causal_graph,
                use_partial_corr,
                rho,
                num_steps,
                num_repeats,
                root_cause_top_k,
            )
        )
        abnormal_metrics = abnormal_metrics + np.random.normal(
            0, 0.01, abnormal_metrics.shape
        )
        result = model.find_root_causes(anomalous_nodes, abnormal_metrics)

        potential_root_causes = [
            PotentialRootCause(root_cause, target_metric, score)
            for root_cause, score in result.root_cause_nodes
        ]
        return potential_root_causes

    return analyze_root_causes
