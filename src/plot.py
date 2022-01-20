from matplotlib.colors import LogNorm
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
        plot_loss_by_max_distance_and_cov_prefactor_ratio_split_by_initialization,
        plot_num_iters_by_max_distance_colored_by_initialization,
        plot_num_clusters_by_max_distance_colored_by_initialization,
        plot_num_initial_clusters_by_max_distance_colored_by_initialization,
        plot_runtime_by_max_distance_colored_by_initialization,
        plot_scores_by_cov_prefactor_ratio_colored_by_initialization,
        plot_scores_by_max_distance_colored_by_initialization,
    ]

    for plot_fn in plot_fns:
        try:
            plot_fn(sweep_results_df=sweep_results_df,
                    plot_dir=plot_dir)
        except Exception as e:
            print(f'Exception: {e}')

        # Close all figure windows to not interfere with next plots
        plt.close('all')


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


def plot_loss_by_max_distance_and_cov_prefactor_ratio_split_by_initialization(
        sweep_results_df: pd.DataFrame,
        plot_dir: str):

    """
    Plots two side-by-side heatmaps of loss (color) by max distance (x) and
    covariance prefactor ratio (y).
    """

    # def draw_heatmap(*args, **kwargs):
    #     data = kwargs.pop('data')
    #     d = data.pivot(index=args[1], columns=args[0], values=args[2])
    #     sns.heatmap(d, **kwargs)
    #
    # fg = sns.FacetGrid(sweep_results_df, col='ini')
    # fg.map_dataframe(draw_heatmap, 'label1', 'label2', 'value', cbar=False)
    #
    # # Make heatmaps square, not rectangular.
    # # See https://stackoverflow.com/questions/41471238/how-to-make-heatmap-square-in-seaborn-facetgrid
    # # get figure background color
    # facecolor = plt.gcf().get_facecolor()
    # for ax in fg.axes.flat:
    #     # set aspect of all axis
    #     ax.set_aspect('equal', 'box-forced')
    #     # set background color of axis instance
    #     ax.set_axis_bgcolor(facecolor)
    # plt.show()

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 8))

    min_loss = sweep_results_df['Loss'].min()
    max_loss = sweep_results_df['Loss'].max()

    for ax_idx, (init_method, sweep_results_subset_df) in enumerate(
            sweep_results_df.groupby('init_method')):

        sweep_results_subset_df = sweep_results_subset_df[
            ['max_distance_param', 'cov_prefactor_ratio', 'Loss']]
        agg_pivot_table = sweep_results_subset_df.pivot_table(
            index='cov_prefactor_ratio',        # y
            columns='max_distance_param',       # x
            values='Loss',                      # z
            aggfunc=np.mean,
            )

        axes[ax_idx].set_title(f'{init_method} Loss')
        sns.heatmap(data=agg_pivot_table,
                    # mask=~np.isnan(agg_pivot_table),
                    ax=axes[ax_idx],
                    vmin=min_loss,
                    vmax=max_loss,
                    square=True,
                    norm=LogNorm())
        axes[ax_idx].set_xlabel(r'$\lambda$')
        axes[ax_idx].set_ylabel(r'$\rho / \sigma$')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir,
                             f'loss_by_max_distance_and_cov_prefactor_ratio_split_by_initialization.png'),
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
        # plt.ylim(0., 1.05)
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
        # plt.ylim(0., 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir,
                                 f'comparison_score={score_column}_by_max_dist.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
