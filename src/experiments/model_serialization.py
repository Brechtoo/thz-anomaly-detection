"""Explizite skops-Freigaben für im Vergleich selbst trainierte Modelle."""

# Aus den vollständigen Fehlerberichten des tsfresh-Laufs abgeleitet.
# Diese internen Typen enthalten trainierten Zustand (Baumknoten, Optimierer
# bzw. native Booster). Nur unsere frisch trainierten Pipelines erhalten
# diese Freigabe; fremde Dateien dürfen damit nicht pauschal geladen werden.
# Keine automatische Übernahme von get_untrusted_types(): Neue Typen müssen
# geprüft und für die betreffende Modellfamilie ausdrücklich ergänzt werden.
_TRUSTED_TYPES = {
    "logistic_regression": (),
    "random_forest": ("sklearn.tree._tree.Tree",),
    "extra_trees": ("sklearn.tree._tree.Tree",),
    "hgb": ("sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",),
    "mlp": ("sklearn.neural_network._stochastic_optimizers.AdamOptimizer",),
    "xgboost": ("xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"),
    "lightgbm": ("collections.OrderedDict", "lightgbm.basic.Booster",
                 "lightgbm.sklearn.LGBMClassifier"),
    "rbf_svm": (),
}


def trusted_types_for_model(name):
    """Frische Liste zurückgeben, damit Aufrufer die Freigaben nicht verändern."""
    return list(_TRUSTED_TYPES[name])
