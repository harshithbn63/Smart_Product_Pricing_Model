import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(200.0 * np.mean(np.abs(y_pred - y_true) / denom))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


