# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Dict, List, Union, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from dowhy import gcm
from dowhy.gcm.auto import AssignmentQuality
from dowhy.gcm.shapley import ShapleyConfig

from rca_task import PotentialRootCause

warnings.filterwarnings("ignore")

gcm.config.disable_progress_bars()


def train_scm(causal_graph: nx.DiGraph,
              normal_metrics: pd.DataFrame) -> Tuple[gcm.InvertibleStructuralCausalModel, pd.DataFrame]:
    scm = gcm.InvertibleStructuralCausalModel(causal_graph)
    gcm.auto.assign_causal_mechanisms(scm, normal_metrics, quality=AssignmentQuality.BETTER)
    gcm.fit(scm, normal_metrics)

    return scm

def reduce_df(df: pd.DataFrame, metric: str, statistic: str):
    data_matrix = df.loc[:, (slice(None), [metric], [statistic])]
    # Now we can focus on the microservices as only column since metric and statistic are fixed.
    data_matrix.columns = [c[0] for c in data_matrix.columns]
    return data_matrix


def pad_and_fill(data_matrix: pd.DataFrame, fill_df: pd.DataFrame):
    original_columns = data_matrix.columns
    overall_mean = np.nanmean(data_matrix.mean())
    for c in fill_df.columns:
        if c not in data_matrix.columns:
            data_matrix[c] = fill_df[c].mean()
    data_matrix.fillna(data_matrix.mean(), inplace=True)
    data_matrix.fillna(overall_mean, inplace=True)
    return data_matrix, original_columns


def pad_and_replace_nan(data_matrix: pd.DataFrame, required_columns: Set[str]):
    # TODO: Cleanup
    data_matrix.fillna(data_matrix.mean(), inplace=True)
    overall_mean = np.nanmean(data_matrix.mean())
    data_matrix.fillna(overall_mean, inplace=True)
    for c in set(required_columns) - set(data_matrix.columns):
        data_matrix[c] = overall_mean
    return data_matrix


def make_counterfactual_attribution_method(
        n_jobs: int=-1,
        attribute_mean_deviation: bool=True,
        anomaly_scorer: gcm.anomaly_scorers.AnomalyScorer=gcm.anomaly_scorers.MeanDeviationScorer()
):
    """Maker of a method to use counterfactuals for root-cause analysis.

        Args:
            n_jobs: use 1 for sequential computation, -1 for parallel.
            attribute_mean_deviation: Indicator whether the contribution is based on the feature relevance with respect
             to the given scoring function or the IT score.
            anomaly_scorer: Anomaly Scorer.
    """
    def analyze_root_causes(
        graph: nx.DiGraph,
        target_node: str,
        target_metric: str,
        target_statistic: str,
        normal_metrics: pd.DataFrame,
        abnormal_metrics: pd.DataFrame,
    ) -> List[PotentialRootCause]:
        """Method to identify potential root causes of a performance issue in target_node through counterfactuals.

        Implementation of Budhathoki et al. "Causal structure based root cause analysis of outliers" in ICML '22.
        https://arxiv.org/abs/1912.02724

        Args:
            graph: Call graph of microservice architecture.
            target_node: Node whose SLO violoation to investigate.
            target_metric: Metric that is in violation with SLO.
            target_statistic: Statistic such as Average of the target_metric that is in violation with SLO.
            normal_metrics: Metrics of all microservices during previous normal operations.
            abnormal_metrics: Metrics of all microservices during SLO violation.

        Returns: List of potential root causes identifying nodes and assigning them scores.
        """
        # We create a simple causal graph by reversing the edges.
        causal_graph = graph.reverse()

        normal_df = pad_and_replace_nan(
            reduce_df(normal_metrics.copy(), metric=target_metric, statistic=target_statistic),
            required_columns=causal_graph.nodes)
        abnormal_df, original_abnormal_columns = pad_and_fill(
            reduce_df(abnormal_metrics.copy(), metric=target_metric, statistic=target_statistic), fill_df=normal_df)
        abnormal_df = abnormal_df.iloc[2:3]
        # Train a structural causal model on normal data for the target_metric and average statistic.
        scm = train_scm(causal_graph, normal_df)
        scores =  gcm.attribute_anomalies(
            scm, target_node=target_node, anomaly_samples=abnormal_df,
            shapley_config=ShapleyConfig(n_jobs=n_jobs),
            attribute_mean_deviation=attribute_mean_deviation,
            anomaly_scorer=anomaly_scorer)
        # Filter out scores for columns for which we had no measurement in abnormal.
        for c in scores:
            if c not in original_abnormal_columns:
                scores[c] = np.zeros(len(scores[c]))
            scores[c][np.isnan(scores[c])] = 0

        return [PotentialRootCause(root_cause, target_metric, np.mean(scores)) for (root_cause, scores) in scores.items()]

    return analyze_root_causes