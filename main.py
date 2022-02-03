import numpy as np
import pandas as pd
import sklearn as sklearn
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from cat_optimize.cat_optimizer import CatOptimizer
from cat_optimize.transformers.naive_cat import NaiveCategories
from evaluate.data_prep.car_prices import CarPrices

from cat_optimize.cat_discover import CatDiscover

pd.set_option("display.max_columns", 10000)
pd.set_option("display.precision", 5)
pd.set_option("expand_frame_repr", False)

def plot_linear_exp(ax, y_true, y_pred):
    y1, y2 = np.exp(y_true), np.exp(y_pred)
    ax.plot(y1, y2, 'o')
    ax.plot((y1.min(), y1.max()), (y1.min(), y1.max()), '-k')

def err_hist(ax, y_true, y_pred):
    y1, y2 = np.exp(y_true), np.exp(y_pred)
    ax.hist(y1 - y2, 20, density=True)

dataset_models = [
    dict(file_path='./evaluate/datasets/Car_Prices_Poland_Kaggle.csv',
         prep=CarPrices(),
         model=RandomForestRegressor(n_estimators=20, min_samples_leaf=5, max_depth=10),
         # model=LinearRegression(),
         metrics=[sklearn.metrics.mean_absolute_error, sklearn.metrics.max_error],
         plots=[plot_linear_exp, err_hist])
]


def experiment_report(name, dataset_params, model, X, Y):
    Y_pred = model.predict(X)
    print()
    print(f'******{name}******')
    print('r2', model.score(X, Y))
    for metric in dataset_params['metrics']:
        print(metric.__name__, metric(Y, Y_pred))
    plots = dataset_params['plots']
    fig, axes = plt.subplots(1, len(plots), figsize=(5 * len(plots), 5))
    for ax, plot in zip(axes.tolist(), plots):
        plot(ax, Y, Y_pred)
        ax.set_title(name)
    plt.show()


def experiment(name, dataset_params, X_train, X_test, Y_train, Y_test):
    model = sklearn.base.clone(dataset_params['model'])
    model.fit(X_train, Y_train)
    experiment_report(f'{name}-train', dataset_params, model, X_train, Y_train)
    experiment_report(f'{name}-test', dataset_params, model, X_test, Y_test)
    errors_train = (Y_train - model.predict(X_train))
    errors_test = (Y_test - model.predict(X_test))
    return errors_train, errors_test


if __name__ == '__main__':
    for dataset in dataset_models:
        data = pd.read_csv(dataset['file_path'])
        X, Y = dataset['prep'].transform(data)
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

        naive_categories = NaiveCategories()
        cat_columns = CatDiscover().discover_categories(X)
        naive_categories.fit_transform(X, cat_columns)
        X_train_naive = naive_categories.transform(X_train)
        X_test_naive = naive_categories.transform(X_test)
        errors_naive_train, errors_naive_test = experiment('NAIVE', dataset, X_train_naive, X_test_naive, Y_train, Y_test)

        auto_categories = CatOptimizer()
        X_train_auto = auto_categories.fit_transform(X_train, np.exp(Y_train))
        X_test_auto = auto_categories.transform(X_test)
        errors_auto_train, errors_auto_test = experiment('AUTO', dataset, X_train_auto, X_test_auto, Y_train, Y_test)

        sns.histplot(data={'naive': np.abs(errors_naive_test.values), 'auto': np.abs(errors_auto_test.values)},
                     legend=['naive', 'auto'], kde=True, )
        plt.show()


