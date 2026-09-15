import copy

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

EPOCHS = 30

DEVICE = ("mps" if torch.backends.mps.is_available() else "cpu")


def predict(model, loader):

    model.eval()

    y_true = []
    y_pred = []
    probabilities = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)

            proba = torch.softmax(logits, dim=1)

            pred = torch.argmax(proba, dim=1)

            y_true.extend(y_batch.numpy())

            y_pred.extend(pred.cpu().numpy())

            probabilities.extend(proba.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(probabilities)


def train_model(model, train_loader, val_loader, loss_function, optimizer):

    best_val_f1 = -np.inf
    best_model = None

    for epoch in range(EPOCHS):

        model.train()

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)

            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = loss_function(predictions, y_batch)

            loss.backward()

            optimizer.step()

        y_val_true, y_val_pred, _ = predict(model, val_loader)

        val_f1 = f1_score(y_val_true, y_val_pred, average="macro")

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_model = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model)

    return model, best_val_f1


def create_loss_function(weight_config, y_train):

    if weight_config is None:

        return (nn.CrossEntropyLoss(), None)

    if weight_config == "balanced":

        weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2, 3]), y=y_train)

    else:

        weights = np.array(weight_config, dtype=np.float32)

    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    return (nn.CrossEntropyLoss(weight=class_weights), weights)