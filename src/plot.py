import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Dict

plt.rcParams.update({'font.size': 20})


def plot_all(sweep_results_df: pd.DataFrame,
             plot_dir: str = 'results'):
    os.makedirs(plot_dir, exist_ok=True)

    plot_fns = [
        plot_loss_by_cov_prefactor_ratio_colored_by_initialization,
        plot_loss_by_max_distance_colored_by_initialization,
        plot_num_iters_by_max_distance_colored_by_initialization,
        plot_num_clusters_by_max_distance_colored_by_initialization,
        plot_num_initial_clusters_by_max_distance_colored_by_initialization,
        plot_runtime_by_max_distance_colored_by_initialization,
        plot_scores_by_cov_prefactor_ratio_colored_by_initialization,
        plot_scores_by_max_distance_colored_by_initialization,
    ]

    for plot_fn in plot_fns:
        plot_fn(sweep_results_df=sweep_results_df,
                plot_dir=plot_dir)


def plot_loss_by_cov_prefactor_ratio_colored_by_initialization(
        sweep_results_df: pd.DataFrame,
        plot_dir: str):

    sns.lineplot(data=sweep_results_df,
                 x='cov_prefactor_ratio',
                 y='Loss',
                 hue='init_method')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\rho / \sigma$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'loss_by_cov_prefactor_ratio.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_loss_by_max_distance_colored_by_initialization(
        sweep_results_df: pd.DataFrame,
        plot_dir: str):

    sns.lineplot(data=sweep_results_df,
                 x='max_distance_param',
                 y='Loss',
                 hue='init_method')
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
        sweep_results_df: pd.DataFrame,
        plot_dir: str):

    sns.lineplot(data=sweep_results_df,
                 x='max_distance_param',
                 y='Num Inferred Clusters',
                 hue='init_method')

    # Can't figure out how to add another line to Seaborn, so manually adding
    # the next line of Num True Clusters.
    num_true_clusters_by_lambda = sweep_results_df[['max_distance_param', 'n_clusters']].groupby('max_distance_param').agg({
        'n_clusters': ['mean', 'sem']
    })['n_clusters']

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
        sweep_results_df: pd.DataFrame,
        plot_dir: str):

    sns.lineplot(data=sweep_results_df,
                 x='max_distance_param',
                 y='Num Initial Clusters',
                 hue='init_method')
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
        sweep_results_df: pd.DataFrame,
        plot_dir: str):

    sns.lineplot(data=sweep_results_df,
                 x='max_distance_param',
                 y='Num Iter Till Convergence',
                 hue='init_method')
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
        sweep_results_df: pd.DataFrame,
        plot_dir: str):

    sns.lineplot(data=sweep_results_df,
                 x='max_distance_param',
                 y='Runtime',
                 hue='init_method')
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


def plot_scores_by_cov_prefactor_ratio_colored_by_initialization(
        sweep_results_df: pd.DataFrame,
        plot_dir: str):

    scores_columns = [col for col in sweep_results_df.columns.values
                      if 'Score' in col]

    for score_column in scores_columns:
        sns.lineplot(data=sweep_results_df,
                     x='cov_prefactor_ratio',
                     y=score_column,
                     hue='init_method')
        plt.xscale('log')
        plt.xlabel(r'$\rho / \sigma$')
        plt.legend()
        plt.ylim(0., 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir,
                                 f'comparison_score={score_column}_by_cov_prefactor_ratio.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_scores_by_max_distance_colored_by_initialization(
        sweep_results_df: pd.DataFrame,
        plot_dir: str):

    scores_columns = [col for col in sweep_results_df.columns.values
                      if 'Score' in col]

    for score_column in scores_columns:
        sns.lineplot(data=sweep_results_df,
                     x='max_distance_param',
                     y=score_column,
                     hue='init_method')
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
