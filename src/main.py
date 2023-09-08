import h5py
from scipy.io import loadmat
import numpy as np
from petsc4py import PETSc
import matplotlib.pyplot as plt

A_file = 'src/L_52x10x22.mat'

# Cargar la matriz A desde el archivo matriz_A.mat
mat_data_A = loadmat(A_file, mat_dtype=True)
matriz_A = mat_data_A['L']

# Cargar el vector b desde el archivo vector_b.mat
mat_data_b = loadmat('src/b_52x10x22.mat', mat_dtype=True)
vector_b = mat_data_b['b']

# Obtener las dimensiones de la matriz A
filas, columnas = matriz_A.shape
print("Dimensiones de la matriz A: ", filas, "x", columnas)

# Crear una matriz PETSc en formato SeqAIJ para matrices dispersas
A = PETSc.Mat().create()
A.setSizes((filas, columnas))
A.setType('seqaij')

# Configurar la matriz con la estructura de la matriz dispersa
A.setUp()
A.setPreallocationNNZ(matriz_A.nnz)

# Establecer los valores de la matriz utilizando setValues
for i in range(filas):
    fila_indices = matriz_A.indices[matriz_A.indptr[i]:matriz_A.indptr[i + 1]]
    fila_valores = matriz_A.data[matriz_A.indptr[i]:matriz_A.indptr[i + 1]]
    A.setValues([i], fila_indices, fila_valores)

# Montar la matriz antes de su uso
A.assemble()

# Crear un vector PETSc para el lado derecho del sistema (b)
b = PETSc.Vec().create()
b.setSizes(filas)
b.setUp()
b.setValues(list(range(filas)), vector_b)

# Crear un vector PETSc para la solución (x)
x = PETSc.Vec().create()
x.setSizes(filas)
x.setUp()

# Crear un objeto KSP (Krylov Subspace Solver) y configurarlo para GMRES
ksp = PETSc.KSP().create()
ksp.setType('gmres')
ksp.setOperators(A)
ksp.setTolerances(rtol=1e-6, atol=1e-12, max_it=1000)
# Crear una lista para almacenar las normas residuales en cada iteración
normas_residuales = []

# Resolver el sistema de ecuaciones Ax = b utilizando GMRES
x = PETSc.Vec().create()
x.setSizes(filas)
x.setUp()

# Configurar la historia de convergencia para almacenar las normas residuales
ksp.setConvergenceHistory()
ksp.solve(b, x)

# Obtener el resultado
solucion = x.getArray()

# Imprimir la solución
print("Solución:")
print(solucion)

# Obtener las normas residuales de la historia de convergencia
normas_residuales = ksp.getConvergenceHistory()

# Graficar la norma residual en función de las iteraciones
iteraciones = np.arange(len(normas_residuales))
print(iteraciones)
plt.figure(figsize=(8, 6))
plt.plot(iteraciones, normas_residuales, marker='o', linestyle='-')
plt.xlabel('Iteración')
plt.ylabel('Norma Residual')
plt.title('Norma Residual en cada Iteración de GMRES')
plt.grid(True)

# Guardar la gráfica en un archivo PNG
plt.savefig('norma_residual.png')