import numpy as np
import pandas as pd


class NaiveCategories:
    """
    Naive transformer of categories into ordinal / sparse space.
    Used for comparison of AutoCat to naive transformations

    Parameters:
        kind - {'ordinal', 'one_hot'} - kind of transformer

    """
    def __init__(self, kind='ordinal'):
        self.kind = kind
        self.category_mapping = {}

    def fit_transform(self, data: pd.DataFrame, columns: list):
        data = data.copy()
        for column in columns:
            if self.kind == 'ordinal':
                data = self._to_ordinal(data, column)
            elif self.kind == 'one_hot':
                data = self._to_one_hot(data, column)
        return data

    def _to_ordinal(self, data, column):
        category_column = data[column].astype('category')
        data[column] = category_column.cat.codes
        self.category_mapping[column] = category_column.cat.categories
        return data

    def _to_one_hot(self, data, column):
        categories = data[column].unique()
        categories = pd.DataFrame(np.eye(len(categories)), index=pd.Series(categories, name=column),
                             columns=[f'{column}={category}' for category in categories])
        categories = categories.reset_index()
        self.category_mapping[column] = categories
        data = data.merge(categories, on=column).drop(columns=[column])
        return data

    def transform(self, data: pd.DataFrame):
        data = data.copy()
        for column, categories in self.category_mapping.items():
            if self.kind == 'ordinal':
                category_column = data[column].astype('category')
                category_column = category_column.cat.set_categories(categories)
                data[column] = category_column.cat.codes
            elif self.kind == 'one_hot':
                data = data.merge(categories, on=column).drop(columns=[column])
        return data

    def reverse(self, data: pd.DataFrame):
        data = data.copy()
        for column, categories in self.category_mapping.items():
            category_column = data[column].map({i: category for i, category in enumerate(categories)})
            data[column] = category_column
        return data
