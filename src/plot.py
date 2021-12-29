import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict


def plot_all(results_df: pd.DataFrame,
             mixture_model_results: Dict,):

    plot_scores_by_max_distance_colored_by_initialization(
        results_df=results_df)

    plot_num_iters_by_max_distance_colored_by_initialization(
        results_df=results_df)

    plot_num_clusters_by_max_distance_colored_by_initialization(
        results_df=results_df,
        true_num_clusters=len(np.unique(mixture_model_results['cluster_assignments'])))


def plot_num_clusters_by_max_distance_colored_by_initialization(
        results_df: pd.DataFrame,
        true_num_clusters: int):

    sns.lineplot(data=results_df, x='lambda', y='Num Clusters',
                 hue='Initialization')
    plt.hlines(y=true_num_clusters,
               xmin=results_df['lambda'].min(),
               xmax=results_df['lambda'].max(),
               label='True')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\lambda$')
    plt.legend()
    plt.show()


def plot_num_iters_by_max_distance_colored_by_initialization(
        results_df: pd.DataFrame):

    sns.lineplot(data=results_df, x='lambda', y='Num Iter To Convergence',
                 hue='Initialization')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\lambda$')
    plt.legend()
    plt.show()


def plot_scores_by_max_distance_colored_by_initialization(
        results_df: pd.DataFrame):

    scores_columns = [col for col in results_df.columns.values
                      if 'Score' in col]

    for score_column in scores_columns:

        sns.lineplot(data=results_df, x='lambda', y=score_column,
                     hue='Initialization')
        plt.xscale('log')
        plt.xlabel(r'$\lambda$')
        plt.legend()
        plt.show()
