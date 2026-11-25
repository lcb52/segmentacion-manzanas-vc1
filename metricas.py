"""Métricas pixel a pixel entre una máscara de referencia y una predicción.

Basado en `compute_metrics` de la Práctica 7. Solo usa NumPy.
Se añaden TP, FP, FN y TN para poder analizar los errores en el notebook.
"""

import numpy as np


def calcular_metricas(ref, resultado):
    """Calcula las métricas de segmentación.

    `ref` y `resultado` son máscaras donde 255 significa "manzana".
    Devuelve un diccionario con los recuentos y las métricas derivadas.
    """
    gt = (ref == 255)
    pred = (resultado == 255)

    TP = int(np.sum(gt & pred))
    TN = int(np.sum(~gt & ~pred))
    FP = int(np.sum(~gt & pred))
    FN = int(np.sum(gt & ~pred))

    return {
        'tp': TP,
        'fp': FP,
        'fn': FN,
        'tn': TN,
        'accuracy':    float((TP + TN) / (TP + TN + FP + FN)),
        'precision':   float(TP / (TP + FP) if (TP + FP) > 0 else 0.0),
        'recall':      float(TP / (TP + FN) if (TP + FN) > 0 else 0.0),
        'fpr':         float(FP / (FP + TN) if (FP + TN) > 0 else 0.0),
        'specificity': float(TN / (TN + FP) if (TN + FP) > 0 else 0.0),
        'iou':         float(TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0),
        'f1':          float(2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0),
    }
