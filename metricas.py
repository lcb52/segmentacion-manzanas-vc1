import numpy as np

def calcular_metricas(ref, resultado):
    gt = (ref == 255)
    pred = (resultado == 255)
    
    TP = np.sum(gt & pred)
    TN = np.sum(~gt & ~pred)
    FP = np.sum(~gt & pred)
    FN = np.sum(gt & ~pred)

    return {
        'accuracy':    float((TP + TN) / (TP + TN + FP + FN)),
        'precision':   float(TP / (TP + FP) if (TP + FP) > 0 else 0.0),
        'recall':      float(TP / (TP + FN) if (TP + FN) > 0 else 0.0),
        'fpr':         float(FP / (FP + TN) if (FP + TN) > 0 else 0.0),
        'specificity': float(TN / (TN + FP) if (TN + FP) > 0 else 0.0),
        'iou':         float(TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0),
        'f1':          float(2*TP / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 0.0)
    }