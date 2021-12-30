import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Dict

plt.rcParams.update({'font.size': 20})


def plot_all(results_df: pd.DataFrame,
             plot_dir: str = 'results'):
    os.makedirs(plot_dir, exist_ok=True)

    plot_fns = [
        plot_loss_by_max_distance_colored_by_initialization,
        plot_num_iters_by_max_distance_colored_by_initialization,
        plot_num_clusters_by_max_distance_colored_by_initialization,
        plot_num_initial_clusters_by_max_distance_colored_by_initialization,
        plot_runtime_by_max_distance_colored_by_initialization,
        plot_scores_by_max_distance_colored_by_initialization,
    ]

    for plot_fn in plot_fns:
        plot_fn(results_df=results_df,
                plot_dir=plot_dir)


def plot_loss_by_max_distance_colored_by_initialization(
        results_df: pd.DataFrame,
        plot_dir: str):

    sns.lineplot(data=results_df, x='lambda', y='Loss',
                 hue='Initialization')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\lambda$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'loss_by_max_dist.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_num_clusters_by_max_distance_colored_by_initialization(
        results_df: pd.DataFrame,
        plot_dir: str):
    sns.lineplot(data=results_df, x='lambda', y='Num Inferred Clusters',
                 hue='Initialization')

    # Can't figure out how to add another line to Seaborn, so manually adding
    # the next line of Num True Clusters.
    num_true_clusters_by_lambda = results_df[['lambda', 'Num True Clusters']].groupby('lambda').agg({
        'Num True Clusters': ['mean', 'sem']
    })['Num True Clusters']

    means = num_true_clusters_by_lambda['mean'].values
    sems = num_true_clusters_by_lambda['sem'].values
    plt.plot(
        num_true_clusters_by_lambda.index.values,
        means,
        label='Num True Clusters',
        color='k',
    )
    plt.fill_between(
        x=num_true_clusters_by_lambda.index.values,
        y1=means - sems,
        y2=means + sems,
        alpha=0.3,
        linewidth=0,
        color='k')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\lambda$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'num_clusters_by_max_dist.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_num_initial_clusters_by_max_distance_colored_by_initialization(
        results_df: pd.DataFrame,
        plot_dir: str):
    sns.lineplot(data=results_df, x='lambda', y='Num Initial Clusters',
                 hue='Initialization')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\lambda$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'num_initial_clusters_by_max_dist.png'),
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
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'num_iters_by_max_dist.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_runtime_by_max_distance_colored_by_initialization(
        results_df: pd.DataFrame,
        plot_dir: str):
    sns.lineplot(data=results_df,
                 x='lambda',
                 y='Runtime',
                 hue='Initialization')
    plt.xscale('log')
    plt.xlabel(r'$\lambda$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'runtime_by_max_dist.png'),
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
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir,
                                 f'comparison_score={score_column}_by_max_dist.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
