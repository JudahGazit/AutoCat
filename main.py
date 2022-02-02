import numpy as np
import pandas as pd
import sklearn as sklearn
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from cat_optimize.transformers.naive_cat import NaiveCategories
from evaluate.data_prep.car_prices import CarPrices

from cat_optimize.cat_discover import CatDiscover

def plot_linear_exp(y_true, y_pred):
    y1, y2 = np.exp(y_true), np.exp(y_pred)
    plt.plot(y1, y2, 'o')
    plt.plot((y1.min(), y1.max()), (y1.min(), y1.max()), '-k')

dataset_models = [
    dict(file_path='./evaluate/datasets/Car_Prices_Poland_Kaggle.csv',
         prep=CarPrices(),
         model=RandomForestRegressor(n_estimators=20, min_samples_leaf=5),
         metrics=[sklearn.metrics.mean_absolute_error, sklearn.metrics.max_error],
         plots=[plot_linear_exp])
]


def experiment_report(name, dataset_params, model, X, Y):
    Y_pred = model.predict(X)
    print('r2', model.score(X, Y))
    for metric in dataset_params['metrics']:
        print(metric.__name__, metric(Y, Y_pred))
    for plot in dataset_params['plots']:
        plot(Y, Y_pred)
        plt.title(name)
        plt.show()


def experiment(dataset_params, X_train, X_test, Y_train, Y_test):
    model = sklearn.base.clone(dataset_params['model'])
    model.fit(X_train, Y_train)
    experiment_report('train', dataset_params, model, X_train, Y_train)
    experiment_report('test', dataset_params, model, X_test, Y_test)


if __name__ == '__main__':
    for dataset in dataset_models:
        data = pd.read_csv(dataset['file_path'])
        X, Y = dataset['prep'].transform(data)

        cat_columns = CatDiscover().discover_categories(X)
        X = NaiveCategories().transform(X, cat_columns)
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

        experiment(dataset, X_train, X_test, Y_train, Y_test)


