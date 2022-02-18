import keras
import keras.layers
import networkx as nx
import pandas as pd
import scipy.stats
import sklearn
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import seaborn as sns

from cat_optimize.cat_discover import CatDiscover


def cramers_V(var1, var2):
    crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))  # Cross table building
    stat = scipy.stats.chi2_contingency(crosstab)[0]  # Keeping of the test statistic of the Chi2 test
    obs = np.sum(crosstab)  # Number of observations
    mini = min(crosstab.shape) - 1  # Take the minimum value between the columns and the rows of the cross table
    return np.sqrt(stat / (obs * mini))


class CatOptimizer:
    def __init__(self, cat_discover=None, min_items_in_cat=0.005, min_categories_to_handle=5, default_cat_name='ELSE'):
        self.cat_discover = cat_discover or CatDiscover()
        self.min_items_in_cat = min_items_in_cat
        self.min_categories_to_handle = min_categories_to_handle
        self.default_cat_name = default_cat_name
        self.category_mapping = {}
        self.category_kind = {}
        self.small_categories_mapping = {}
        self.onehot_embeddings = {}

    def _convert_to_categorical(self, data, categorical_columns):
        for column in categorical_columns:
            category_column = data[column].astype('category')
            data[column] = category_column
            self.category_mapping[column] = category_column.cat.categories
        return data

    def _corr_of_categories(self, categories: pd.Series, target: pd.Series, is_classification):
        if is_classification:
            # corr = {c: cramers_V((categories == c).astype(int), target.astype(int)) for c in
            #         categories.unique()}
            # return pd.Series(corr)
            ctab = pd.crosstab(categories, target, normalize='index')
            points = ctab.values
            g = nx.minimum_spanning_tree(nx.from_numpy_array(scipy.spatial.distance.cdist(points, points)))
            g_sort = list(nx.bfs_tree(g, min(g, key=lambda k: g.degree(k))))
            return pd.Series(np.arange(len(g_sort)), index=categories.cat.categories[g_sort])
        else:
            one_hot = pd.get_dummies(categories)
            return one_hot.corrwith(target)

    def _find_small_categories(self, data, target, is_classification):
        min_items_for_category = self.min_items_in_cat if isinstance(self.min_items_in_cat,
                                                                     int) else self.min_items_in_cat * len(data)
        for column in self.category_mapping:
            value_counts = data[column].value_counts()
            convert_to_else = value_counts[value_counts < min_items_for_category].index.tolist()
            if len(value_counts) > self.min_categories_to_handle and len(convert_to_else) > 3:
                effect_on_target = self._mean_target_per_category(data, column, target, is_classification)
                effect_on_target = effect_on_target.loc[convert_to_else]
                _, p = scipy.stats.shapiro(effect_on_target.sample(min(len(effect_on_target), 30)))
                if p < 0.05 and len(convert_to_else) > 10:
                    new_categories = self.__split_into_several_bins(column, convert_to_else, effect_on_target)
                else:
                    new_categories = [category for category in self.category_mapping[column] if
                                      category not in convert_to_else] + [self.default_cat_name]
                self.category_mapping[column] = pd.Index(new_categories)

    def __split_into_several_bins(self, column, convert_to_else, effect_on_target, n_bins=1):
        if n_bins > 1:
            clusters = pd.qcut(effect_on_target, n_bins, labels=range(n_bins))
            self.small_categories_mapping[column] = dict(zip(effect_on_target.index, clusters))
            new_categories = [category for category in self.category_mapping[column] if category not in convert_to_else] + \
                             [f'{self.default_cat_name}_{i}' for i in np.unique(clusters)]
            return new_categories
        return self.category_mapping[column]

    def _convert_missing_categories(self, data):
        for column in self.category_mapping:
            if column in self.small_categories_mapping:
                data[column] = data[column].apply(lambda v: v if v in self.category_mapping[column]
                else f'{self.default_cat_name}_{self.small_categories_mapping[column].get(v, 0)}')
            else:
                data[column] = data[column].apply(
                    lambda v: v if v in self.category_mapping[column] else self.default_cat_name)
            data[column] = data[column].astype('category').cat.set_categories(self.category_mapping[column])
        return data

    def _find_and_sort_categories(self, data, target, is_classification):
        for column in self.category_mapping:
            # mean_target = self._mean_target_per_category(data, column, target, is_classification)
            self.category_kind[column] = 'ordinal'
            corr = self._sort_category_by_correlation(column, data, target, is_classification)
            if abs(corr) < 0.05:
                self.category_kind[column] = 'one_hot'
                self._build_one_hot_encoder(data, column, target, is_classification)
            # if len(mean_target) <= 10 or scipy.stats.shapiro(mean_target.sample(min(len(mean_target), 30)))[1] < 0.05:
            #     self.category_kind[column] = 'ordinal'
            #     self._sort_category_by_correlation(column, data, target, is_classification)
            # else:
            #     self.category_kind[column] = 'one_hot'
            #     self._build_one_hot_encoder(data, column, target)
        return data

    def _mean_target_per_category(self, data, column, target, is_classification):
        data_with_target = data.join(target)
        if is_classification:
            ctab = pd.crosstab(data[column], target)
            mean_target = ((ctab - ctab.mean()) / ctab.std()).mean(1)
        else:
            mean_target = data_with_target.groupby(column).mean()[target.name]
        return mean_target

    def _sort_category_by_correlation(self, column, data, target, is_classification):
        correlation_to_target = self._corr_of_categories(data[column], target, is_classification)
        sorted_categories = correlation_to_target.sort_values().index.to_list()
        self.category_mapping[column] = sorted_categories
        data[column] = data[column].cat.reorder_categories(sorted_categories)
        if is_classification:
            return cramers_V(data[column].cat.codes, target.astype(int))
        return data[column].cat.codes.corr(target)

    def _build_one_hot_encoder(self, data, column, target, is_classification):
        num_categories = len(data[column].cat.categories)
        loss = 'categorical_crossentropy' if is_classification else 'mse'
        if num_categories > 16 and False:
            model = keras.models.Sequential()
            model.add(keras.layers.Input(1))
            model.add(keras.layers.Embedding(num_categories, 16))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(target.nunique() if is_classification else 1,
                                         activation='softmax' if is_classification else 'linear'))
            model.compile(optimizer='adam', loss=loss, metrics=['accuracy' if is_classification else 'mse'])
            intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=model.layers[0].output)
            model.fit(data[column].cat.codes, np.eye(target.nunique())[target.values.astype(int)], 32, 2, True)
            embedding = intermediate_layer_model.predict(np.arange(num_categories)).squeeze()
        else:
            embedding = np.eye(num_categories)[:, 1:]
        self.onehot_embeddings[column] = embedding

    def _convert_categories_to_numeric(self, data, column, target=None):
        data[column] = data[column].cat.codes
        return data

    def _convert_to_one_hot(self, data, column):
        encodings = self.onehot_embeddings[column][data[column].cat.codes]
        encodings = pd.DataFrame(encodings, columns=[f'{column}_{i}' for i in range(encodings.shape[1])],
                                 index=data.index)
        data = data.drop(columns=[column]).join(encodings)
        return data

    def _convert_by_kind(self, data):
        convertors = dict(one_hot=self._convert_to_one_hot, ordinal=self._convert_categories_to_numeric)
        for column, kind in self.category_kind.items():
            data = convertors[kind](data, column)
        return data

    def fit_transform(self, data: pd.DataFrame, target: pd.Series):
        data = data.copy()
        # is_classification = False
        is_classification = target.dtype in (np.int64, np.int, np.int32) or target.apply(float.is_integer).all()
        categorical_columns = self.cat_discover.discover_categories(data)
        # if target.dtype in (np.int64, np.int, np.int32) or target.apply(float.is_integer).all():
        #     target = data[data.corrwith(target).abs().sort_values().index[-1]].rename('target')
        data = self._convert_to_categorical(data, categorical_columns)
        self._find_small_categories(data, target, is_classification)
        data = self._convert_missing_categories(data)
        data = self._find_and_sort_categories(data, target, is_classification)
        data = self._convert_by_kind(data)
        print(data.head())
        return data

    def transform(self, data: pd.DataFrame):
        data = self._convert_missing_categories(data)
        data = self._convert_by_kind(data)
        return data
