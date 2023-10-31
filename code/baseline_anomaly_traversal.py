# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Any, Tuple, Union, Callable, Optional, Set

import networkx as nx
import numpy as np
import pandas as pd

from dowhy.gcm import graph


from rca_task import PotentialRootCause
from data_preprocessing import (
    AnomalyDetectionConfig,
    get_anomalous_metrics_and_scores,
    DEFAULT_ANOMALY_DETECTION,
)


def _anomaly_traversal(
    causal_graph: nx.DiGraph, anomaly_nodes: List[Any]
) -> Dict[str, Any]:
    """Traverses the graph and returns the nodes without anomalous parent.

    :param causal_graph: Graph to analyze.
    :param anomaly_nodes: A list of the names of anomalous nodes.
    :return: A dictionary with the root causes as keys and NAN as score.
    """
    try:
        graph.validate_acyclic(causal_graph.subgraph(anomaly_nodes))
    except RuntimeError:
        raise ValueError(
            "The subgraph containing the anomalous nodes has to be acyclic for the traversal algorithm"
            "but the subgraph of anomalous nodes is cyclic!"
        )
    results = {}

    for anomaly_node in anomaly_nodes:
        parents = causal_graph.predecessors(anomaly_node)
        if not set(anomaly_nodes) & set(parents):
            results[anomaly_node] = np.nan

    return results


def _score_potential_root_causes(
    causal_graph: nx.DiGraph,
    target_node: str,
    metric: str,
    root_causes: Dict[str, float],
    all_anomalies: Dict[str, float],
) -> Dict[str, float]:
    """Returns all the paths from the root causes to the target node.

    :param causal_graph: Graph to analyze.
    :param target_node: The name of the target node.
    :param root_causes: A dictionary containing the root causes with scores.
    :param all_anomalies: A dictionary containing the scores of all anomalous nodes.
    :return: A list that contains all the paths from the root causes to the
             target node and the second entry is a dictionary.
    """
    results = {}

    for root_cause in root_causes:
        paths = list(nx.all_simple_paths(causal_graph, root_cause, target_node))
        for path in map(nx.utils.pairwise, paths):
            path = list(path)
            is_anomalous_path = True
            nodes_on_path = set({})
            for edge in path:
                is_anomalous_path &= (
                    edge[0] in all_anomalies and edge[1] in all_anomalies
                )
                nodes_on_path.update([edge[0], edge[1]])

            if is_anomalous_path:
                score = np.sum([all_anomalies[n] for n in nodes_on_path])
                results[root_cause] = max(results.get(root_cause, -np.inf), score)
    return [
        PotentialRootCause(root_cause, metric, score)
        for (root_cause, score) in results.items()
    ]


def _traversal_rca(
    causal_graph: nx.DiGraph,
    target_node: str,
    target_metric: str,
    normal_metrics: pd.DataFrame,
    abnormal_metrics: pd.DataFrame,
    anomaly_detection_config: AnomalyDetectionConfig,
    statistic_of_interest: str = "Average",
    search_for_anomaly: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Performs anomaly detection and traverses the causal graph to identify the root caues.

    :param causal_graph: A causal graph.
    :param target_node: Node whose SLO violoation to investigate.
    :param target_metric: Metric that is in violation with SLO.
    :param normal_metrics: Metrics of all microservices during previous normal operations.
    :param abnormal_metrics: Metrics of all microservices during SLO violation.
    :param anomaly_detection_config: The configuration for the anomaly detection.
    :param statistic_of_interest: The desired statistic for each metric. Must be in
            {'Sum','Average','p50','p90','p95','p99'}.
    :param search_for_anomaly: A boolean indicating whether to search for anomalies in the causal graph.
    :return: A tuple containing two dictionaries, where the first one contains the root causes with scores and the
             second one all anomalous nodes with their scores.
    """
    all_anomalous_nodes_with_score = get_anomalous_metrics_and_scores(
        normal_metrics,
        abnormal_metrics,
        target_node,
        target_metric,
        anomaly_detection_config,
        statistic_of_interest,
        search_for_anomaly,
    )
    if all_anomalous_nodes_with_score == ({}, []):
        return all_anomalous_nodes_with_score
    root_causes = _anomaly_traversal(
        causal_graph,
        list(
            {
                c[0]
                for c in abnormal_metrics.columns
                if c[0] in all_anomalous_nodes_with_score.keys()
            }
        ),
    )

    # The traversal algorithm doesn't return a score, i.e., we need to set it.
    for rc in root_causes:
        root_causes[rc] = all_anomalous_nodes_with_score[rc]

    return root_causes, all_anomalous_nodes_with_score


def make_baseline_analyze_root_causes(
    anomaly_detection_config: AnomalyDetectionConfig = DEFAULT_ANOMALY_DETECTION,
):
    def baseline_analyze_root_causes(
        graph: nx.DiGraph,
        target_node: str,
        target_metric: str,
        target_statistic: str,
        normal_metrics: pd.DataFrame,
        abnormal_metrics: pd.DataFrame,
    ) -> List[PotentialRootCause]:
        """Method to identify potential root causes that of a performance issue in target_node.

        Args:
            graph: Call graph of microservice architecture.
            target_node: Node whose SLO violation to investigate.
            target_metric: Metric that is in violation with SLO.
            normal_metrics: Metrics of all microservices during previous normal operations.
            abnormal_metrics: Metrics of all microservices during SLO violation.

        Returns: List of potential root causes identifying nodes and assigning them scores.
        """
        # We create a simple causal graph by reversing the edges.
        causal_graph = graph.reverse()

        # Apply traversal algorithm which identifies anomalous nodes without anomalous parents.
        root_causes, all_anomalous_nodes = _traversal_rca(
            causal_graph,
            target_node,
            target_metric,
            normal_metrics,
            abnormal_metrics,
            anomaly_detection_config,
            statistic_of_interest=target_statistic,
        )
        # Filter to root causes with anomalous paths to target and sum up scores along the way.
        return _score_potential_root_causes(
            causal_graph, target_node, target_metric, root_causes, all_anomalous_nodes
        )

    return baseline_analyze_root_causes
