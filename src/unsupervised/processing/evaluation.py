import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_anomaly_scores(y_true, scores, top_k_values):

    results = pd.DataFrame({
        "label": y_true.to_numpy(),
        "score": scores
    })

    # --------------------------------------------------------
    # label 1 vs 3
    # --------------------------------------------------------

    binary = results[results["label"].isin([1, 3])].copy()

    # label 1 = positive klasse
    y_binary = (binary["label"] == 1).astype(int)

    roc_auc = roc_auc_score(y_binary, binary["score"])

    average_precision = (average_precision_score(y_binary, binary["score"]))

    baseline_ap = y_binary.mean()

    # --------------------------------------------------------
    # top k
    # --------------------------------------------------------

    ranked = results.sort_values("score", ascending=False)

    metrics = {
        "roc_auc": roc_auc,
        "average_precision":average_precision,
        "baseline_ap": baseline_ap
    }

    for k in top_k_values:

        top = ranked.head(k)

        label1_count = (top["label"] == 1).sum()

        label12_count = (top["label"].isin([1, 2])).sum()

        metrics[f"top_{k}_label1"] = label1_count / k

        metrics[f"top_{k}_label12"] = label12_count / k

    return metrics