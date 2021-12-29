import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Dict


plt.rcParams.update({'font.size': 22})


def plot_all(results_df: pd.DataFrame,
             plot_dir: str = 'results'):

    os.makedirs(plot_dir, exist_ok=True)

    plot_scores_by_max_distance_colored_by_initialization(
        results_df=results_df,
        plot_dir=plot_dir)

    plot_num_iters_by_max_distance_colored_by_initialization(
        results_df=results_df,
        plot_dir=plot_dir)

    plot_num_clusters_by_max_distance_colored_by_initialization(
        results_df=results_df,
        plot_dir=plot_dir)


def plot_num_clusters_by_max_distance_colored_by_initialization(
        results_df: pd.DataFrame,
        plot_dir: str):

    sns.lineplot(data=results_df, x='lambda', y='Num Inferred Clusters',
                 hue='Initialization')
    # plt.hlines(y=true_num_clusters,
    #            xmin=results_df['lambda'].min(),
    #            xmax=results_df['lambda'].max(),
    #            label='True')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\lambda$')
    plt.legend()
    plt.savefig(os.path.join(plot_dir,
                             f'num_clusters_by_max_dist.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_num_iters_by_max_distance_colored_by_initialization(
        results_df: pd.DataFrame,
        plot_dir: str):

    sns.lineplot(data=results_df, x='lambda', y='Num Iter Till Convergence',
                 hue='Initialization')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\lambda$')
    plt.legend()
    plt.savefig(os.path.join(plot_dir,
                             f'num_iters_by_max_dist.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_scores_by_max_distance_colored_by_initialization(
        results_df: pd.DataFrame,
        plot_dir: str):

    scores_columns = [col for col in results_df.columns.values
                      if 'Score' in col]

    for score_column in scores_columns:

        sns.lineplot(data=results_df, x='lambda', y=score_column,
                     hue='Initialization')
        plt.xscale('log')
        plt.xlabel(r'$\lambda$')
        plt.legend()
        plt.ylim(0., 1.05)
        plt.savefig(os.path.join(plot_dir,
                                 f'comparison_score={score_column}_by_max_dist.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()

