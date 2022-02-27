import random

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_friedman1

from evaluate.data_prep.data_prep_base import DataPrepBase


class GenerateDataset(DataPrepBase):
    """
    Synthetic dataset generator using builtin sklearn generation methods (from `sklearn.datasets`).
    To create categorical features, a random subset of features is drawn and either binned into categories or being
    assigned with random categorical labels.

    Parameters:
        :param data_type: {'regression', 'linear_regression', 'classification'} - type of dataset generated.
            `regression` uses `make_friedman1`
            `linear_regression` uses `make_regression`
            `classification` uses `make_classification`
        :param n_rows: number of rows to generate
        :param n_features: number of features to generate
        :param n_classes: number of output classes to generate, only in used when `data_type == "classification"`
        :param n_categorical: number of categorical features
        :param min_categories: minimal number of categories to extract from each feature
        :param max_categories: maximal number of categories to extract from each feature
        :param n_informative: number of informative features in dataset
        :param noise: noise to apply, only in use when `data_type == "regression"`
        :param ordinal_prob: probability to create ordinal feature, default 0.8

    """
    def __init__(self, data_type='regression',
                 n_rows=10000, n_features=10, n_classses=2,
                 n_categorical=8, min_categories=2, max_categories=20,
                 n_informative=8, noise=0.1, ordinal_prob=0.8):
        self.data_type = data_type
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_classses = n_classses
        self.n_categorical = n_categorical
        self.min_categories = min_categories
        self.max_categories = max_categories
        self.n_informative = n_informative
        self.noise = noise
        self.ordinal_prob = ordinal_prob

    def _convert_to_categories(self, data):
        categorical_columns = np.random.choice(self.n_features, self.n_categorical, False)
        n_categories = np.random.randint(self.min_categories, self.max_categories, self.n_categorical)
        data = pd.DataFrame(data, columns=[f'c{i}_f' if i not in categorical_columns else f'c{i}_cat' for i in range(data.shape[1])])
        for column, n_bins in zip(categorical_columns, n_categories):
            column_name = f'c{column}_cat'
            labels = np.random.permutation([str(i) for i in range(n_bins)])
            if random.random() < self.ordinal_prob:
                data[column_name] = pd.cut(data[column_name], n_bins, labels=labels).astype(str)
            else:
                data[column_name] = np.random.choice(labels, len(data))
        return data

    def transform(self, data=None) -> (pd.DataFrame, pd.DataFrame):
        if self.data_type == 'regression':
            X, Y = make_friedman1(self.n_rows, self.n_features, noise=self.noise)
        elif self.data_type == 'linear_regression':
            X, Y = make_regression(self.n_rows, self.n_features,
                                   n_informative=self.n_informative, noise=self.noise)
        else:
            X, Y = make_classification(self.n_rows, self.n_features, n_classes=self.n_classses,
                                       n_informative=self.n_informative)
        X = self._convert_to_categories(X)
        Y = pd.Series(Y, name='target')
        return X, Y
