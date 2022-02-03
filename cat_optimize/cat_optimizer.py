import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import seaborn as sns

from cat_optimize.cat_discover import CatDiscover


class CatOptimizer:
    def __init__(self, cat_discover=None, min_items_in_cat=0.005, min_categories_to_handle=5, default_cat_name='ELSE'):
        self.cat_discover = cat_discover or CatDiscover()
        self.min_items_in_cat = min_items_in_cat
        self.min_categories_to_handle = min_categories_to_handle
        self.default_cat_name = default_cat_name
        self.category_mapping = {}
        self.category_kind = {}
        self.small_categories_mapping = {}

    def _convert_to_categorical(self, data, categorical_columns):
        for column in categorical_columns:
            category_column = data[column].astype('category')
            data[column] = category_column
            self.category_mapping[column] = category_column.cat.categories
        return data

    def __corr_of_categories(self, categories: pd.Series, target: pd.Series):
        one_hot = pd.get_dummies(categories)
        corrs = []
        for column in one_hot.columns:
            corrs.append(target.corr(one_hot[column]))
        return pd.Series(corrs, index=one_hot.columns)

    def _find_small_categories(self, data, target):
        data_with_target = data.join(target)
        min_items_for_category = self.min_items_in_cat if isinstance(self.min_items_in_cat, int) else self.min_items_in_cat * len(data)
        for column in self.category_mapping:
            value_counts = data[column].value_counts()
            convert_to_else = value_counts[value_counts < min_items_for_category].index.tolist()
            if len(value_counts) > self.min_categories_to_handle and len(convert_to_else) > 1:
                effect_on_target = data_with_target.groupby(column).mean().loc[convert_to_else][target.name]
                _, p = scipy.stats.shapiro(effect_on_target)
                if p < 0.05 and len(convert_to_else) > 10:
                    new_categories = self.__split_into_several_bins(column, convert_to_else, effect_on_target)
                else:
                    new_categories = [category for category in self.category_mapping[column] if category not in convert_to_else] + [self.default_cat_name]
                self.category_mapping[column] = pd.Index(new_categories)

    def __split_into_several_bins(self, column, convert_to_else, effect_on_target):
        clusters = pd.qcut(effect_on_target, 3, labels=range(3))
        self.small_categories_mapping[column] = dict(zip(effect_on_target.index, clusters))
        new_categories = [category for category in self.category_mapping[column] if category not in convert_to_else] + \
                         [f'{self.default_cat_name}_{i}' for i in np.unique(clusters)]
        return new_categories

    def _convert_missing_categories(self, data):
        for column in self.category_mapping:
            if column in self.small_categories_mapping:
                data[column] = data[column].apply(lambda v: v if v in self.category_mapping[column]
                else f'{self.default_cat_name}_{self.small_categories_mapping[column].get(v, 0)}')
            else:
                data[column] = data[column].apply(lambda v: v if v in self.category_mapping[column] else self.default_cat_name)
            data[column] = data[column].astype('category').cat.set_categories(self.category_mapping[column])
        return data

    def _find_and_sort_categories(self, data, target):
        data_with_target = data.join(target)
        for column in self.category_mapping:
            mean_target = data_with_target.groupby(column).mean()[target.name]
            _, p_value = scipy.stats.shapiro(mean_target)
            if p_value < 0.05:
                self.category_kind[column] = 'ordinal'
                self.__sort_category_by_correlation(column, data, target)
            else:
                self.category_kind[column] = 'one_hot'
        return data

    def __sort_category_by_correlation(self, column, data, target):
        correlation_to_target = self.__corr_of_categories(data[column], target)
        sorted_categories = correlation_to_target.sort_values().index.to_list()
        self.category_mapping[column] = sorted_categories
        data[column] = data[column].cat.reorder_categories(sorted_categories)

    def _convert_categories_to_numeric(self, data, column):
        data[column] = data[column].cat.codes
        return data

    def _convert_to_one_hot(self, data, column):
        one_hot = pd.get_dummies(data[column])
        one_hot = one_hot.rename(columns={category: f'{column}={category}' for category in one_hot.columns})
        data = data.drop(columns=[column]).join(one_hot)
        return data

    def _convert_by_kind(self, data):
        for column, kind in self.category_kind.items():
            if kind == 'one_hot':
                data = self._convert_to_one_hot(data, column)
            else:
                data = self._convert_categories_to_numeric(data, column)
        return data

    def fit_transform(self, data: pd.DataFrame, target: pd.Series):
        data = data.copy()
        categorical_columns = self.cat_discover.discover_categories(data)
        data = self._convert_to_categorical(data, categorical_columns)
        self._find_small_categories(data, target)
        data = self._convert_missing_categories(data)
        data = self._find_and_sort_categories(data, target)
        data = self._convert_by_kind(data)
        return data

    def transform(self, data: pd.DataFrame):
        data = self._convert_missing_categories(data)
        data = self._convert_by_kind(data)
        return data