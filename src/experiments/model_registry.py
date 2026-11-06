
"""Zentrale Modellkonfiguration mit bewusst einfachen Standard-Hyperparametern."""

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ============================================================
# 1. MODELLE AKTIVIEREN / DEAKTIVIEREN
# ============================================================

# Legt fest, welche Modelle trainiert werden sollen.
MODEL_ENABLED = {
    "logistic_regression": True,
    "random_forest": True,
    "hgb": True,
    "xgboost": True,
    "lightgbm": True,
    "extra_trees": True,
    "mlp": True,
    "rbf_svm": False,
}

# Diese Modelle profitieren von standardisierten Eingabefeatures.
SCALED_MODELS = {"logistic_regression", "mlp", "rbf_svm"}


# ============================================================
# 2. EIN MODELL MIT PREPROCESSING ERSTELLEN
# ============================================================

def create_model(name, seed=42):
    """
    Erstellt eine vollständige sklearn-Pipeline für das gewählte Modell.

    name: Name des Modells
    seed: Zufallsstartwert für möglichst reproduzierbare Ergebnisse

    Rückgabe: Pipeline aus Preprocessing und Klassifikationsmodell.
    """

    # Dictionary mit Modellnamen und zugehörigen Erzeugungsfunktionen.
    #
    # lambda erstellt eine anonyme Funktion.
    # Das Modell wird dadurch erst beim Aufruf erzeugt.
    #
    # Beispiel:
    # factories["random_forest"]()
    # erzeugt eine neue RandomForestClassifier-Instanz.

    factories = {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=1000,  # Maximale Anzahl Optimierungsiterationen.
            random_state=seed,
        ),

        "random_forest": lambda: RandomForestClassifier(
            random_state=seed,
            n_jobs=1,  # Ein paralleler Job innerhalb des Modells.
        ),

        "hgb": lambda: HistGradientBoostingClassifier(
            random_state=seed,
        ),

        "extra_trees": lambda: ExtraTreesClassifier(
            random_state=seed,
            n_jobs=1,
        ),

        "mlp": lambda: MLPClassifier(
            random_state=seed,
            max_iter=200,
        ),

        "rbf_svm": lambda: SVC(
            kernel="rbf",  # Nichtlinearer RBF-Kernel.
            random_state=seed,
        ),
    }

    # XGBoost und LightGBM werden nur importiert,
    # wenn das jeweilige Modell tatsächlich ausgewählt wurde.
    #
    # -> Nicht benötigte Bibliotheken müssen nicht geladen werden.

    if name == "xgboost":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            random_state=seed,
            n_jobs=1,
        )

    elif name == "lightgbm":
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            random_state=seed,
            n_jobs=1,
            verbosity=-1,  # Weniger Konsolenausgaben.
        )

    else: estimator = factories[name]()

    # ========================================================
    # 3. PREPROCESSING-PIPELINE AUFBAUEN
    # ========================================================

    # tsfresh hat bereits imputiert und Features selektiert.
    # Deshalb hier keine erneute Imputation.

    # Erster Pipeline-Schritt:
    # Features ohne Varianz entfernen.
    #
    # VarianceThreshold() verwendet standardmäßig threshold=0.
    # Entfernt also Spalten, deren Werte alle identisch sind.
    #
    # Die Entscheidung wird später beim fit() anhand
    # der Trainingsdaten getroffen.

    steps = [("constants", VarianceThreshold())]

    # Falls das gewählte Modell Skalierung benötigt:
    if name in SCALED_MODELS:

        # StandardScaler als weiteren Verarbeitungsschritt hinzufügen.
        #
        # Er berechnet beim Training Mittelwert und Standardabweichung
        # jedes Features und verwendet diese später auch für neue Daten.
        steps.append(("scale", StandardScaler()))

    # Das eigentliche Klassifikationsmodell als letzten Schritt hinzufügen.
    steps.append(("model", estimator))

    # Die Pipeline enthält beispielsweise für Logistic Regression:
    #
    # 1. VarianceThreshold
    # 2. StandardScaler
    # 3. LogisticRegression
    #
    # Beim späteren pipeline.fit(X_train, y_train)
    # werden die Schritte automatisch in dieser Reihenfolge ausgeführt.
    #
    # Beim pipeline.predict(X_val) werden dieselben
    # trainierten Transformationen angewendet.

    return Pipeline(steps)


# ============================================================
# 4. ALLE AKTIVIERTEN MODELLE ERSTELLEN
# ============================================================

def build_models(enabled=None, seed=42):
    """
    Erstellt ein Dictionary mit allen aktivierten Modell-Pipelines.

    enabled:
        Optionales Dictionary mit Modellschaltern.
        Falls None, wird MODEL_ENABLED verwendet.

    seed:
        Zufallsstartwert für die Modelle.
    """

    # Entweder die übergebene Konfiguration verwenden
    # oder auf die Standardkonfiguration zurückgreifen.
    switches = MODEL_ENABLED if enabled is None else enabled

    # Überprüfen, ob unbekannte Modellnamen übergeben wurden.
    #
    # set(switches) enthält die Namen aus der Konfiguration.
    # Die Mengendifferenz liefert alle nicht unterstützten Namen.
    unknown = set(switches) - set(MODEL_ENABLED)

    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    # Dictionary Comprehension:
    # Für jedes aktivierte Modell wird create_model() aufgerufen.
    #
    # Beispiel:
    # {
    #     "random_forest": Pipeline(...),
    #     "hgb": Pipeline(...),
    #     "mlp": Pipeline(...),
    # }
    #
    # Hier wird noch NICHT trainiert, sondern nur initialisiert.

    models = {
        name: create_model(name, seed)
        for name, active in switches.items()
        if active
    }

    # Verhindern, dass ein Experiment ohne aktive Modelle startet.
    if not models:
        raise ValueError("Enable at least one model")

    # Dictionary mit sämtlichen aktivierten Pipelines zurückgeben.
    return models
