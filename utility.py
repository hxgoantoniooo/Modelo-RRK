import pandas as pd
import numpy as np

# Carga de datos con o sin configuración
def load_data_csv(data_file, config_file=None):
    data = pd.read_csv(data_file, header=None).to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]
    Y = -Y
    if config_file:
        with open(config_file) as f:
            sigma2 = float(f.readline())
            lambd = float(f.readline())
        return X, Y, sigma2, lambd
    return X, Y

# Kernel Gaussiano
def kernel_mtx(X1, X2, sigma2):
    A = np.sum(X1**2, axis=1).reshape(-1, 1)
    B = np.sum(X2**2, axis=1).reshape(1, -1)
    C = X1 @ X2.T
    dist2 = A + B - 2 * C
    return np.exp(-dist2 / (2 * sigma2))

# Cálculo de coeficientes beta
def krr_coeff(K, y, lamb):    
    n = K.shape[0]
    return np.linalg.pinv(K + lamb * np.eye(n)) @ y

# Matriz de confusión (2x2)
def confusion_mtx(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    cm[0,0] = np.sum((y_true == 1)  & (y_pred == 1))  # TP
    cm[0,1] = np.sum((y_true == -1) & (y_pred == 1))  # FP
    cm[1,0] = np.sum((y_true == 1)  & (y_pred == -1)) # FN
    cm[1,1] = np.sum((y_true == -1) & (y_pred == -1)) # TN
    return cm

# Métricas: f-score por clase (columna 2x1)
def metricas(y_true, y_pred):
    cm = confusion_mtx(y_true, y_pred)
    TP, FP = cm[0,0], cm[0,1]
    FN, TN = cm[1,0], cm[1,1]

    prec1 = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec1  = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) > 0 else 0

    prec2 = TN / (TN + FN) if (TN + FN) > 0 else 0
    rec2  = TN / (TN + FP) if (TN + FP) > 0 else 0
    f2 = 2 * prec2 * rec2 / (prec2 + rec2) if (prec2 + rec2) > 0 else 0

    return np.array([[f1], [f2]])
