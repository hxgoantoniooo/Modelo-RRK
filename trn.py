# Training of RRK-Pinv
import numpy as np
from utility import *

def main():
    # Step 1: Cargar datos
    Xtrn, Ytrn, sigma2, lambd = load_data_csv('dtrain.csv', 'config.csv')

    # Step 2: Kernel Gaussiano entre datos de entrenamiento
    Ker = kernel_mtx(Xtrn, Xtrn, sigma2)

    # Step 3: Calcular coeficientes beta
    beta = krr_coeff(Ker, Ytrn, lambd)

    # Step 4: Guardar beta
    np.savetxt('beta.csv', beta, delimiter=',')

if __name__ == "__main__":
    main()
