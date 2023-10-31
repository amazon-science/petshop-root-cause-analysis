# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from pyrca.analyzers.ht import HT, HTConfig

import networkx as nx
import pandas as pd

from rca_task import PotentialRootCause
from data_preprocessing import reduce_df, marginalize_node, impute_df


def make_circa(
    aggregator: str = "max",
    root_cause_top_k: int = 3,
    adjustment: bool = True,
    imputation_method: str = "mean",
):
    """
    Wrapper around the 'Hypothesis Testing-based' RCA method, CIRCA, as implemented in https://github.com/salesforce/PyRCA.

    Paper: https://dl.acm.org/doi/10.1145/3534678.3539041

    Args:
        aggregator: The function for aggregating the node score from all the abnormal data.
        root_cause_top_k: The maximum number of root causes in the results.
        adjustment: Whether to perform descendant adjustment.
        imputation_method: How NaNs should be imputed. If 'mean' then each is replaced by the mean of the
            remaining values of the same microservice, metric and statistic. If 'interpolate' then
            pandas.DataFrame.interpolate(method='time',limit_direction='both') is used. Default: 'mean'
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

        normal_metrics = reduce_df(normal_metrics, target_metric, statistic_of_interest)
        normal_metrics = normal_metrics.loc[
            :, normal_metrics.columns[~normal_metrics.isna().all()]
        ]

        abnormal_metrics = reduce_df(
            abnormal_metrics, target_metric, statistic_of_interest
        )
        abnormal_metrics = abnormal_metrics.loc[
            :, abnormal_metrics.columns[~abnormal_metrics.isna().all()]
        ]

        common_cols = normal_metrics.columns.intersection(abnormal_metrics.columns)
        normal_metrics = normal_metrics.loc[:, common_cols]
        abnormal_metrics = abnormal_metrics.loc[:, common_cols]

        impute_df(normal_metrics, imputation_method)
        impute_df(abnormal_metrics, imputation_method)

        missing_nodes = normal_metrics.columns.symmetric_difference(
            causal_graph.nodes()
        )

        for node in missing_nodes:
            marginalize_node(causal_graph, node)

        causal_graph = nx.to_pandas_adjacency(causal_graph)

        model = HT(config=HTConfig(causal_graph, aggregator, root_cause_top_k))
        model.train(normal_metrics)
        result = model.find_root_causes(abnormal_metrics, target_node, adjustment)

        potential_root_causes = [
            PotentialRootCause(root_cause, target_metric, score)
            for root_cause, score in result.root_cause_nodes
        ]
        return potential_root_causes

    return analyze_root_causes
