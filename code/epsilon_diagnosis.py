# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from pyrca.analyzers.epsilon_diagnosis import EpsilonDiagnosis

import networkx as nx
import pandas as pd

from rca_task import PotentialRootCause
from data_preprocessing import map_df, impute_df


def make_epsilon_diagnosis(
    alpha: float = 0.05,
    bootstrap_times: int = 200,
    root_cause_top_k: int = 3,
    limit_metric: bool = True,
    imputation_method: str = "mean",
):
    """
    Wrapper around epsilon-diagnosis RCA method as implemented in https://github.com/salesforce/PyRCA.

    Paper: https://dl.acm.org/doi/10.1145/3308558.3313653

    Args:
        alpha: The desired significance level (float) in (0, 1). Default: 0.05.
        bootstrap_times: Bootstrap times.
        root_cause_top_k: The maximum number of root causes in the results.
        limit_metric: Whether to restrict RCA to metrics of the same type as that of the detected fault.
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
        if limit_metric:
            normal_metrics = normal_metrics.loc[
                :, (slice(None), [target_metric], [statistic_of_interest])
            ]
            abnormal_metrics = abnormal_metrics.loc[
                :, (slice(None), [target_metric], [statistic_of_interest])
            ]
        else:
            # Assuming here that it doesn't make sense to *not* limit the statistic
            normal_metrics = normal_metrics.loc[
                :, (slice(None), slice(None), [statistic_of_interest])
            ]
            abnormal_metrics = abnormal_metrics.loc[
                :, (slice(None), slice(None), [statistic_of_interest])
            ]

        normal_metrics = map_df(normal_metrics)
        abnormal_metrics = map_df(abnormal_metrics)

        # remove columns for which all values are missing
        normal_metrics = normal_metrics.loc[
            :, normal_metrics.columns[~normal_metrics.isna().all()]
        ]
        abnormal_metrics = abnormal_metrics.loc[
            :, abnormal_metrics.columns[~abnormal_metrics.isna().all()]
        ]

        common_cols = normal_metrics.columns.intersection(abnormal_metrics.columns)
        normal_metrics = normal_metrics.loc[:, common_cols]
        abnormal_metrics = abnormal_metrics.loc[:, common_cols]

        impute_df(normal_metrics, imputation_method)
        impute_df(abnormal_metrics, imputation_method)

        model = EpsilonDiagnosis(
            config=EpsilonDiagnosis.config_class(
                alpha, bootstrap_times, root_cause_top_k
            )
        )
        # Take the same number of (most recent) normal values as is available in the abnormal regime
        model.train(normal_metrics.tail(abnormal_metrics.shape[0]))
        result = model.find_root_causes(abnormal_metrics.tail(normal_metrics.shape[0]))

        potential_root_causes = []
        for root_cause, score in result.root_cause_nodes:
            name = root_cause.split("_")
            metric = name[-2]
            stat = name[-1]
            node = "_".join(name[:-2])
            potential_root_causes.append(
                PotentialRootCause(node=node, metric=metric, score=score)
            )
        return potential_root_causes

    return analyze_root_causes
