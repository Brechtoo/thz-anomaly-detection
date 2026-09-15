from pathlib import Path

import numpy as np
import torch

from src.evaluation.metrics import evaluate_predictions, calculate_error_counts

from src.models.transformer import TransformerModel
from src.paths import TRANSFORMER_RESULTS

from src.transformer.training import DEVICE, create_loss_function, train_model, predict


LEARNING_RATE = 1e-3

OUTPUT_DIR = TRANSFORMER_RESULTS

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_configuration(name, weight_config, train_loader, val_loader, y_train, val_df):

    np.random.seed(42)

    torch.manual_seed(42)

    # --------------------------------------------------------
    # loss
    # --------------------------------------------------------

    loss_function, weights = (create_loss_function(weight_config, y_train))

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    model = TransformerModel().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --------------------------------------------------------
    # train
    # --------------------------------------------------------

    model, best_val_f1 = train_model(model, train_loader, val_loader, loss_function, optimizer)

    # --------------------------------------------------------
    # pred
    # --------------------------------------------------------

    y_true, y_pred, proba = predict(model, val_loader)

    # --------------------------------------------------------
    # eval
    # --------------------------------------------------------

    metrics = evaluate_predictions(y_true, y_pred)

    error_counts = calculate_error_counts(y_true, y_pred)

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------

    results = val_df.copy()

    results["y_pred"] = y_pred

    for label in range(4):
        results[f"proba_{label}"] = proba[:, label]

    results.to_csv(OUTPUT_DIR / f"transformer_{name}.csv", index=False)

    torch.save(model.state_dict(), OUTPUT_DIR / f"transformer_{name}.pt")

    return {
        "configuration": name,
        **metrics,
        **error_counts,
        "best_val_macro_f1": best_val_f1,
    }