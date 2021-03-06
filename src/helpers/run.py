import pandas as pd
import wandb


def download_wandb_project_runs_results(wandb_project_path: str,
                                        sweep_id: str = None,
                                        ) -> pd.DataFrame:

    # Download sweep results
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    if sweep_id is None:
        runs = api.runs(path=wandb_project_path)
    else:
        runs = api.runs(path=wandb_project_path,
                        filters={"Sweep": sweep_id})

    sweep_results_list = []
    for run in runs:

        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary = run.summary._json_dict

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        summary.update(
            {k: v for k, v in run.config.items()
             if not k.startswith('_')})

        summary.update({'State': run.state,
                        'Sweep': run.sweep.id if run.sweep is not None else None})
        # .name is the human-readable name of the run.
        summary.update({'run_name': run.name})
        sweep_results_list.append(summary)

    sweep_results_df = pd.DataFrame(sweep_results_list)

    # Keep only finished runs
    finished_runs = sweep_results_df['State'] == 'finished'
    print(f"% of successfully finished runs: {finished_runs.mean()}")
    sweep_results_df = sweep_results_df[finished_runs]

    sweep_results_df = sweep_results_df.copy()

    return sweep_results_df
