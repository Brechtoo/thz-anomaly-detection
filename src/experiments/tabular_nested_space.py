"""Fixed search designs for the five non-SVM tabular families."""

from itertools import product

MODELS = ('mlp', 'logistic_regression', 'lightgbm', 'hgb', 'xgboost')
COSTS = {
    'uniform': None,
    'balanced': 'balanced',
    'cavity_cost': {0: 1., 1: 2.5, 2: 1.6, 3: .85},
}

# Each row changes a complementary set of capacity and shrinkage controls.
PROFILES = {
    'mlp': [
        {'hidden_layer_sizes': (48,), 'alpha': .001, 'learning_rate_init': .001},
        {'hidden_layer_sizes': (96,), 'alpha': .01, 'learning_rate_init': .001},
        {'hidden_layer_sizes': (64, 24), 'alpha': .0001, 'learning_rate_init': .001},
        {'hidden_layer_sizes': (128, 48), 'alpha': .001, 'learning_rate_init': .0005},
        {'hidden_layer_sizes': (48, 24), 'alpha': .01, 'learning_rate_init': .003},
    ],
    'logistic_regression': [
        {'C': .003}, {'C': .03}, {'C': .3}, {'C': 3.}, {'C': 30.},
    ],
    'lightgbm': [
        {'num_leaves': 9, 'min_child_samples': 45, 'learning_rate': .04, 'n_estimators': 130},
        {'num_leaves': 17, 'min_child_samples': 30, 'learning_rate': .06, 'n_estimators': 130},
        {'num_leaves': 31, 'min_child_samples': 25, 'learning_rate': .04, 'n_estimators': 150,
         'reg_lambda': 1.},
        {'num_leaves': 47, 'min_child_samples': 55, 'learning_rate': .08, 'n_estimators': 120,
         'reg_lambda': 3.},
        {'num_leaves': 23, 'min_child_samples': 15, 'learning_rate': .09, 'n_estimators': 90,
         'colsample_bytree': .7, 'subsample': .8, 'subsample_freq': 1},
    ],
    'hgb': [
        {'max_leaf_nodes': 9, 'min_samples_leaf': 45, 'learning_rate': .05, 'max_iter': 120},
        {'max_leaf_nodes': 17, 'min_samples_leaf': 30, 'learning_rate': .08, 'max_iter': 120},
        {'max_leaf_nodes': 31, 'min_samples_leaf': 25, 'learning_rate': .05, 'max_iter': 135,
         'l2_regularization': 1.},
        {'max_leaf_nodes': 47, 'min_samples_leaf': 55, 'learning_rate': .09, 'max_iter': 105,
         'l2_regularization': 3.},
        {'max_leaf_nodes': 23, 'min_samples_leaf': 15, 'learning_rate': .11, 'max_iter': 80,
         'max_features': .7},
    ],
    'xgboost': [
        {'max_depth': 2, 'min_child_weight': 8, 'learning_rate': .07, 'n_estimators': 130},
        {'max_depth': 3, 'min_child_weight': 5, 'learning_rate': .09, 'n_estimators': 110},
        {'max_depth': 4, 'min_child_weight': 3, 'learning_rate': .05, 'n_estimators': 150},
        {'max_depth': 6, 'min_child_weight': 8, 'learning_rate': .08, 'n_estimators': 110,
         'reg_lambda': 3.},
        {'max_depth': 5, 'min_child_weight': 2, 'learning_rate': .09, 'n_estimators': 95,
         'colsample_bytree': .7, 'subsample': .8},
    ],
}
BASE = {'hgb': {'early_stopping': False}}


def grid(model):
    fixed = BASE.get(model, {})
    return [
        {'candidate': 'reference', 'weights': 'uniform', 'params': fixed},
        *[
            {'candidate': f'p{profile}_{weight}', 'weights': weight, 'params': fixed | params}
            for (profile, params), weight in product(enumerate(PROFILES[model], 1), COSTS)
        ],
    ]
