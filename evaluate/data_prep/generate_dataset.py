import random

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.datasets import make_classification, make_regression, make_friedman1

from evaluate.data_prep.data_prep_base import DataPrepBase


class GenerateDataset(DataPrepBase):
    def __init__(self, data_type='regression',
                 n_rows=10000, n_features=20, n_classses=2,
                 n_categorical=10, min_categories=2, max_categories=20,
                 n_informative=10, noise=0.1):
        self.data_type = data_type
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_classses = n_classses
        self.n_categorical = n_categorical
        self.min_categories = min_categories
        self.max_categories = max_categories
        self.n_informative = n_informative
        self.noise = noise

    def _convert_to_categories(self, data, target):
        categorical_columns = np.random.choice(self.n_features, self.n_categorical, False)
        n_categories = np.random.randint(self.min_categories, self.max_categories, self.n_categorical)
        data = pd.DataFrame(data, columns=[f'c{i}_f' if i not in categorical_columns else f'c{i}_cat' for i in range(data.shape[1])])
        for column, n_bins in zip(categorical_columns, n_categories):
            column_name = f'c{column}_cat'
            labels = np.random.permutation([str(i) for i in range(n_bins)])
            if random.random() < 0.8:
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
        X = self._convert_to_categories(X, Y)
        Y = pd.Series(Y, name='target')
        return X, Y


if __name__ == '__main__':
    pd.set_option("display.max_columns", 10000)
    pd.set_option("display.precision", 5)
    pd.set_option("expand_frame_repr", False)

    print(GenerateDataset().transform())
