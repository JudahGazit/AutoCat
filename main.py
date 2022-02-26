import numpy as np
import pandas as pd
import scipy.stats
import sklearn as sklearn
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from pmlb import fetch_data
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBRegressor, XGBClassifier

from cat_optimize.cat_optimizer import CatOptimizer
from cat_optimize.transformers.naive_cat import NaiveCategories
from evaluate.data_prep.australian_weather import AustralianWeather
from evaluate.data_prep.car_prices import CarPrices

from cat_optimize.cat_discover import CatDiscover
from evaluate.data_prep.cat_challenge import CatChallenge
from evaluate.data_prep.generate_dataset import GenerateDataset
from evaluate.data_prep.insurance import Insurance
from evaluate.data_prep.pmlb_datasets import PMLDBDatasets
from evaluate.data_prep.strokes import Strokes
from evaluate.data_prep.student_grades import StudentGrades

pd.set_option("display.max_columns", 10000)
pd.set_option("display.precision", 5)
pd.set_option("expand_frame_repr", False)

def plot_linear_exp(ax, y_true, y_pred, f=np.exp):
    y1, y2 = f(y_true), f(y_pred)
    ax.plot(y1, y2, 'o')
    ax.plot((y1.min(), y1.max()), (y1.min(), y1.max()), '-k')

def err_hist(ax, y_true, y_pred, f=np.exp):
    y1, y2 = f(y_true), f(y_pred)
    ax.hist(y1 - y2, 20, density=True)

dataset_models = [
    dict(name='car_prices',
         data=pd.read_csv('./evaluate/datasets/Car_Prices_Poland_Kaggle.csv').sample(10000),
         prep=CarPrices(),
         model=[RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_depth=10),
                LinearRegression(),
                XGBRegressor(max_depth=5, n_estimators=20)],
         metrics=[sklearn.metrics.mean_absolute_error, sklearn.metrics.max_error],
         plots=[plot_linear_exp, err_hist],
         target_method=np.exp),

    dict(name='synthetic_regression',
         data=None,
         prep=GenerateDataset(data_type='regression'),
         model=[
             RandomForestRegressor(n_estimators=20, min_samples_leaf=5, max_depth=10),
             LinearRegression(),
             XGBRegressor(max_depth=5, n_estimators=20)],
         metrics=[sklearn.metrics.mean_absolute_error, sklearn.metrics.max_error],
         plots=[lambda *args, **kwargs: plot_linear_exp(f=lambda x: x, *args, **kwargs),
                lambda *args, **kwargs: err_hist(f=lambda x: x, *args, **kwargs)]),

    dict(name='synthetic_linear_regression',
         data=None,
         prep=GenerateDataset(data_type='linear_regression'),
         model=[
             RandomForestRegressor(n_estimators=20, min_samples_leaf=5, max_depth=10),
             LinearRegression(),
             XGBRegressor(max_depth=5, n_estimators=20)],
         metrics=[sklearn.metrics.mean_absolute_error, sklearn.metrics.max_error],
         plots=[lambda *args, **kwargs: plot_linear_exp(f=lambda x: x, *args, **kwargs),
                lambda *args, **kwargs: err_hist(f=lambda x: x, *args, **kwargs)]),

    # dict(name='synthetic_classification',
    #      data=None,
    #      prep=GenerateDataset(data_type='classification', n_classses=2),
    #      model=[RandomForestClassifier(n_estimators=20, min_samples_leaf=1, max_depth=20),
    #             LogisticRegression(max_iter=500),
    #             XGBClassifier(max_depth=20, n_estimators=20)
    #             ],
    #      metrics=(sklearn.metrics.accuracy_score,
    #               sklearn.metrics.precision_score,
    #               sklearn.metrics.recall_score,
    #               sklearn.metrics.f1_score
    #               ),
    #      plots=[]),

    # dict(name='australian_weather',
    #      data=pd.read_csv('./evaluate/datasets/weatherAUS.csv'),
    #      prep=AustralianWeather(),
    #      model=[RandomForestClassifier(n_estimators=20, min_samples_leaf=3, max_features=0.5, max_depth=10, class_weight='balanced'),
    #             LogisticRegression(max_iter=500, class_weight='balanced'),
    #             XGBClassifier(max_depth=10, n_estimators=10, scale_pos_weight=80)
    #             ],
    #      metrics=[sklearn.metrics.accuracy_score, sklearn.metrics.precision_score, sklearn.metrics.recall_score, sklearn.metrics.f1_score],
    #      plots=[]),
    #
    # dict(name='strokes',
    #      data=pd.read_csv('./evaluate/datasets/healthcare-dataset-stroke-data.csv'),
    #      prep=Strokes(),
    #      model=[RandomForestClassifier(n_estimators=20, max_depth=50, class_weight='balanced', ),
    #             LogisticRegression(max_iter=500),
    #             XGBClassifier(max_depth=50, n_estimators=20, scale_pos_weight=80)],
    #      metrics=[sklearn.metrics.precision_score, sklearn.metrics.recall_score, sklearn.metrics.f1_score],
    #      plots=[]),
]


def experiment_report(name, dataset_params, model, X, Y):
    Y_pred = model.predict(X)
    metric_results = dict(r2=model.score(X, Y), **{metric.__name__: metric(Y, Y_pred) for metric in dataset_params['metrics']})
    # print()
    # print(f'******{name}******')
    # print(metric_results)
    # plots = dataset_params['plots']
    # if len(plots):
    #     fig, axes = plt.subplots(1, len(plots), figsize=(5 * len(plots), 5))
    #     for ax, plot in zip(axes.tolist(), plots):
    #         plot(ax, Y, Y_pred)
    #         ax.set_title(name)
    #     plt.show()
    return metric_results


def experiment(name, dataset_params, X_train, X_test, Y_train, Y_test):
    model = sklearn.base.clone(dataset_params['model'])
    model.fit(X_train, Y_train)
    metric_results_train = experiment_report(f'{name}-train', dataset_params, model, X_train, Y_train)
    metric_results_test = experiment_report(f'{name}-test', dataset_params, model, X_test, Y_test)
    errors_train = (Y_train - model.predict(X_train))
    errors_test = (Y_test - model.predict(X_test))

    # try:
    #     explainer = shap.Explainer(model)
    #     shap_values = explainer.shap_values(X_test[:100])
    #     shap.summary_plot(shap_values, X_test[:100])
    # except: pass
    metric_results = pd.DataFrame([pd.Series(metric_results_train, name='train'), pd.Series(metric_results_test, name='test')]).transpose()
    return errors_train, errors_test, metric_results


def experiment_on_dataset(dataset):
    data = dataset['data']
    X, Y = dataset['prep'].transform(data)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
    naive_categories = NaiveCategories()
    cat_columns = CatDiscover().discover_categories(X)
    naive_categories.fit_transform(X, cat_columns)
    X_train_naive = naive_categories.transform(X_train)
    X_test_naive = naive_categories.transform(X_test)
    errors_naive_train, errors_naive_test, naive_results = experiment('NAIVE', dataset, X_train_naive, X_test_naive,
                                                                      Y_train, Y_test)
    auto_categories = CatOptimizer()
    X_train_auto = auto_categories.fit_transform(X_train, Y_train if dataset.get('target_method') is None else dataset[
        'target_method'](Y_train))
    X_test_auto = auto_categories.transform(X_test)
    errors_auto_train, errors_auto_test, auto_results = experiment('AUTO', dataset, X_train_auto, X_test_auto, Y_train,
                                                                   Y_test)
    # sns.histplot(data={'naive': np.abs(errors_naive_test.values), 'auto': np.abs(errors_auto_test.values)},
    #              legend=['naive', 'auto'], kde=True, )
    # plt.show()
    df = naive_results.join(auto_results, lsuffix='_naive', rsuffix='_auto')
    df['train_impr'] = df['train_auto'] / df['train_naive']
    df['test_impr'] = df['test_auto'] / df['test_naive']
    # print(df)
    #
    # print('KS', scipy.stats.ks_2samp(np.abs(errors_naive_test.values), np.abs(errors_auto_test.values)))
    # print('chi2', scipy.stats.chisquare(errors_auto_test.abs().value_counts(), errors_naive_test.abs().value_counts()))
    return df


if __name__ == '__main__':
    dataframes = []
    for dataset in dataset_models:
        # print(experiment_on_dataset())
        for model in dataset['model']:
            dataset_of_model = dataset.copy()
            dataset_of_model['model'] = model
            dfs = [experiment_on_dataset(dataset_of_model).reset_index() for _ in range(20)]
            d = pd.concat(dfs).groupby('index').agg(list)
            d['pvalue'] = d[['test_naive', 'test_auto']].apply(lambda x: scipy.stats.ttest_1samp(np.array(x[1]) / np.array(x[0]), 1)[1], axis=1)
            df = pd.concat(dfs).groupby('index').agg(['mean', 'std'])
            df.columns = ['_'.join(c) for c in df.columns.to_flat_index()]
            df = df.join(d.pvalue)
            df['dataset'] = dataset_of_model['name']
            df['model'] = model.__class__.__name__
            print(df)
            dataframes.append(df)

    dataframes = pd.concat(dataframes)
    print(dataframes)
    # dataframes.to_excel('data.xlsx')


