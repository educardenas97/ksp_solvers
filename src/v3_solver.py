import scipy.io
import numpy as np
from scipy.sparse.linalg import gmres
import logging

# Configura el sistema de registro
logging.basicConfig(filename='solver_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Carga el archivo .mat
mat = scipy.io.loadmat('src/sherman5_2.mat')

# Extrae la matriz A y el vector b del archivo .mat
A = mat['A']
b = mat['b']

# Convierte la matriz A a un formato compatible con SciPy
A = A.tocsr()  # Si A es una matriz dispersa, conviértela a formato CSR

# Define una función para registrar los eventos del solver
def solver_callback(xk):
    logging.info(f'Iteración {solver_callback.iteration}: ||A*x - b|| = {np.linalg.norm(A.dot(xk) - b)}')

    solver_callback.iteration += 1

solver_callback.iteration = 1

# Configura el solucionador GMRES con el callback
x, _ = gmres(A, b, callback=solver_callback)

# Registra información sobre la solución final
logging.info(f'Solución final x: {x}')
