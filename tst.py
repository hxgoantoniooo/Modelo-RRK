# Testing of RRK for IDS
import numpy as np
from utility import *

def main():
    # Step 1: Cargar datos de test
    Xtst, Ytst = load_data_csv('dtest.csv')
    Xtrn, Ytrn, sigma2, lamb = load_data_csv('dtrain.csv', 'config.csv')
    print(sigma2, lamb)
    beta = np.loadtxt('beta.csv', delimiter=',')

    # Step 2: Kernel Gaussiano entre datos de test y entrenamiento
    Ker_test = kernel_mtx(Xtst, Xtrn, sigma2)

    # Step 3: Predicción z = Ker * beta
    z = Ker_test @ beta

    # Step 4: Clasificación binaria: -1 / +1
    y_pred = np.where(z >= 0, 1, -1)

    # Step 5: Matriz de confusión y f-scores
    cm = confusion_mtx(Ytst, y_pred)
    print(cm)
    fs = metricas(Ytst, y_pred)

    # Step 6: Guardar resultados
    np.savetxt('cmatriz.csv', cm, fmt='%d', delimiter=',')
    np.savetxt('fscores.csv', fs, fmt='%.4f', delimiter=',')

if __name__ == "__main__":
    main()
