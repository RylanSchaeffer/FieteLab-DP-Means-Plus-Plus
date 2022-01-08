import numpy as np
import os
import pandas as pd
import sklearn
import torch
import torchvision
from typing import Callable, Dict, List, Tuple, Union


def load_dataset(dataset_name: str,
                 dataset_kwargs: Dict = None,
                 data_dir: str = 'data',
                 ) -> Dict[str, np.ndarray]:

    if dataset_name == 'boston_housing_1993':
        load_dataset_fn = load_boston_housing_1993
    elif dataset_name == 'cancer_gene_expression_2016':
        load_dataset_fn = load_cancer_gene_expression_2016
    # elif dataset_name == 'covid_hospital_treatment_2020':
    #     load_dataset_fn = load_covid_hospital_treatment_2020
    elif dataset_name == 'diabetes_hospitals_2014':
        load_dataset_fn = load_diabetes_hospitals_2014
    elif dataset_name == 'electric_grid_stability_2016':
        load_dataset_fn = load_electric_grid_stability_2016
    elif dataset_name == 'wisconsin_breast_cancer_1995':
        load_dataset_fn = load_wisconsin_breast_cancer_1995
    else:
        raise NotImplementedError
    dataset_dict = load_dataset_fn(
        data_dir=data_dir,
        **dataset_kwargs)
    return dataset_dict


def load_boston_housing_1993(data_dir: str = 'data',
                             **kwargs,
                             ) -> Dict[str, np.ndarray]:
    """
    Properties:
      - dtype: Real, Binary
      - Samples: 506
      - Dimensions: 12
      - Link: https://www.kaggle.com/arslanali4343/real-estate-dataset
    """
    dataset_dir = os.path.join(data_dir,
                               'boston_housing_1993')
    observations_path = os.path.join(dataset_dir, 'data.csv')

    data = pd.read_csv(observations_path, index_col=False)
    observations = data.loc[:, ~data.columns.isin(['MEDV'])]

    # MEDV Median value of owner-occupied homes in $1000's
    # Without rounding, there are 231 classes. With rounding, there are 48.
    labels = data['MEDV'].round().astype('category').cat.codes

    dataset_dict = dict(
        observations=observations.values,
        labels=labels.values,
    )

    return dataset_dict


def load_cancer_gene_expression_2016(data_dir: str = 'data',
                                     **kwargs,
                                     ) -> Dict[str, np.ndarray]:
    """

    Properties:
      - dtype: Real
      - Samples: 801
      - Dimensions: 20531
      - Link: https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq#
    """

    dataset_dir = os.path.join(data_dir,
                               'cancer_gene_expression_2016')
    observations_path = os.path.join(dataset_dir, 'data.csv')
    labels_path = os.path.join(dataset_dir, 'labels.csv')
    observations = pd.read_csv(observations_path, index_col=0)
    labels = pd.read_csv(labels_path, index_col=0)

    # Convert strings to integer codes
    labels['Class'] = labels['Class'].astype('category').cat.codes

    # Exclude any row containing any NaN
    obs_rows_with_nan = observations.isna().any(axis=1)
    label_rows_with_nan = observations.isna().any(axis=1)
    rows_without_nan = ~(obs_rows_with_nan | label_rows_with_nan)
    observations = observations[rows_without_nan]
    labels = labels[rows_without_nan]

    dataset_dict = dict(
        observations=observations.values,
        labels=labels.values,
    )

    return dataset_dict


def load_covid_hospital_treatment_2020(data_dir: str = 'data',
                                       **kwargs,
                                       ) -> Dict[str, np.ndarray]:
    """
    Most of these are categorical - not good. Return to later

    :param data_dir:
    :return:
    """

    dataset_dir = os.path.join(data_dir,
                               'covid_hospital_treatment_2020')
    data_path = os.path.join(dataset_dir, 'host_train.csv')

    data = pd.read_csv(data_path, index_col=False)
    observations = data.loc[:, ~data.columns.isin(['id', 'diagnosis'])]
    labels = data['Stay_Days'].astype('category').cat.codes

    dataset_dict = dict(
        observations=observations.values,
        labels=labels.values,
    )

    return dataset_dict


def load_diabetes_hospitals_2014(data_dir: str = 'data',
                                 **kwargs,
                                 ) -> Dict[str, np.ndarray]:
    dataset_dir = os.path.join(data_dir,
                               'diabetes_hospitals_2014')
    data_path = os.path.join(dataset_dir, 'data.csv')

    data = pd.read_csv(data_path, index_col=False)
    observations = data.loc[:, ~data.columns.isin(['id', 'diagnosis'])]
    labels = data['diagnosis'].astype('category').cat.codes

    dataset_dict = dict(
        observations=observations.values,
        labels=labels.values,
    )

    return dataset_dict


def load_electric_grid_stability_2016(data_dir: str = 'data',
                                      **kwargs,
                                      ) -> Dict[str, np.ndarray]:
    dataset_dir = os.path.join(data_dir,
                               'electric_grid_stability_2016')
    data_path = os.path.join(dataset_dir, 'smart_grid_stability_augmented.csv')

    data = pd.read_csv(data_path, index_col=False)
    observations = data.loc[:, ~data.columns.isin(['stab', 'stabf'])]

    # Rather than using binary 'stabf' as the class, use deciles (arbitrarily chosen)
    labels = pd.qcut(data['stab'], 10).astype('category').cat.codes

    dataset_dict = dict(
        observations=observations.values,
        labels=labels.values,
    )

    return dataset_dict


def load_template(data_dir: str = 'data',
                  **kwargs,
                  ) -> Dict[str, np.ndarray]:
    dataset_dir = os.path.join(data_dir,
                               'wisconsin_breast_cancer_1995')
    data_path = os.path.join(dataset_dir, 'data.csv')

    data = pd.read_csv(data_path, index_col=False)
    observations = data.loc[:, ~data.columns.isin(['id', 'diagnosis'])]
    labels = data['diagnosis'].astype('category').cat.codes

    dataset_dict = dict(
        observations=observations.values,
        labels=labels.values,
    )

    return dataset_dict


def load_wisconsin_breast_cancer_1995(data_dir: str = 'data',
                                      **kwargs,
                                      ) -> Dict[str, np.ndarray]:
    """
    Properties:
      - dtype: Real
      - Samples: 569
      - Dimensions: 32
      - Link: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
      - Data: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/version/2

    :param data_dir:
    :return:
    """
    dataset_dir = os.path.join(data_dir,
                               'wisconsin_breast_cancer_1995')
    data_path = os.path.join(dataset_dir, 'data.csv')
    data = pd.read_csv(data_path, index_col=False)

    observations = data.loc[:, ~data.columns.isin(['id', 'diagnosis'])]
    labels = data['diagnosis'].astype('category').cat.codes

    dataset_dict = dict(
        observations=observations.values,
        labels=labels.values,
    )

    return dataset_dict


if __name__ == '__main__':
    load_dataset(dataset_name='electric_grid_stability_2016')
