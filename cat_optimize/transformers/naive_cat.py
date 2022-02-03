import pandas as pd


class NaiveCategories:
    def __init__(self):
        self.category_mapping = {}

    def fit_transform(self, data: pd.DataFrame, columns: list):
        data = data.copy()
        for column in columns:
            category_column = data[column].astype('category')
            data[column] = category_column.cat.codes
            self.category_mapping[column] = category_column.cat.categories
        return data

    def transform(self, data: pd.DataFrame):
        data = data.copy()
        for column, categories in self.category_mapping.items():
            category_column = data[column].astype('category')
            category_column = category_column.cat.set_categories(categories)
            data[column] = category_column.cat.codes
        return data

    def reverse(self, data: pd.DataFrame):
        data = data.copy()
        for column, categories in self.category_mapping.items():
            category_column = data[column].map({i: category for i, category in enumerate(categories)})
            data[column] = category_column
        return data
