import numpy as np
from utility import *

def difference_fs(tg_c1, tg_c2, c1, c2):
    diff_c1 = abs(tg_c1 - c1)
    diff_c2 = abs(tg_c2 - c2)
    return diff_c1, diff_c2


# Cargar datos
Xtrn, Ytrn, _, _ = load_data_csv('dtrain.csv', 'config.csv')
Xtst, Ytst = load_data_csv('dtest.csv')

# Matriz objetivo
target_cm = np.array([[126, 11],
                      [5,   120]])

target_c1, target_c2 = 0.9403, 0.9375


# Rangos a probar (ajústalos si quieres)
sigma2 = 4
lambda_vals = np.arange(0.01, 1 + 0.001, 0.01)
print(lambda_vals)    # 30 valores entre 1e-4 y 1
best_lambda = 0
best_c1diff = 0
best_c2diff = 0
best_diff = abs(best_c1diff - best_c2diff)

# Búsqueda
for lambd in lambda_vals:
    print(f"Sigma^2: {sigma2}, Lambda: {lambd}")
    # Entrenar
    K = kernel_mtx(Xtrn, Xtrn, sigma2)
    beta = krr_coeff(K, Ytrn, lambd)
    # Predecir
    Ktest = kernel_mtx(Xtst, Xtrn, sigma2)
    z = Ktest @ beta
    y_pred = np.where(z >= 0, 1, -1)
    # Evaluar
    cm = confusion_mtx(Ytst, y_pred)
    c1, c2, fs = metricas(Ytst, y_pred)
    diffc1, diffc2 = difference_fs(target_c1, target_c2, c1, c2)
    print(c1, c2)
    print(diffc1, diffc2)
    diff_abs = abs(diffc1 - diffc2)
    if best_c1diff == 0 and best_c2diff == 0:
        best_c1diff, best_c2diff = diffc1, diffc2
        best_lambda = lambd
    elif best_c1diff > diffc1 and best_c2diff > diffc2 and best_diff > diff_abs:
        best_c1diff, best_c2diff = diffc1, diffc2
        best_lambda = lambd
    print(cm)
    # Comparar
    if np.array_equal(cm, target_cm):
        print("✅ ¡Encontrado!")
        print(f"Sigma² = {sigma2:.6f}, Lambda = {lambd:.6f}")
        print("Confusion matrix:")
        print(cm)
        exit()

print("No se encontró ninguna combinación exacta.")
print(f"Lambda: {best_lambda}, Clase1: {best_c1diff}, Clase2: {best_c2diff}")
