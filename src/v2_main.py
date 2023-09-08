import h5py
from scipy.io import loadmat
import numpy as np
from petsc4py import PETSc
import matplotlib.pyplot as plt
import os

A_file = 'src/sherman5_2.mat'
matrix_name = os.path.splitext(os.path.basename(A_file))[0]
solver_name = 'gmres'  # Cambia el nombre del solver según el método utilizado

# Cargar la matriz A desde el archivo matriz_A.mat
mat_data_A = loadmat(A_file, mat_dtype=True)
matriz_A = mat_data_A['A']

# Cargar el vector b desde el archivo vector_b.mat
mat_data_b = loadmat('src/sherman5_2.mat', mat_dtype=True)
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
ksp.setType(solver_name)
ksp.setOperators(A)
# Establecer el número máximo de iteraciones
max_iteraciones = 10000  # Puedes ajustar este valor según tus necesidades
ksp.setTolerances(rtol=1e-6, atol=1e-8, max_it=max_iteraciones)
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
# Obtener las normas residuales de la historia de convergencia
normas_residuales = ksp.getConvergenceHistory()

# Graficar la norma residual en función de las iteraciones de manera logarítmica
iteraciones = np.arange(len(normas_residuales))
log_iteraciones = np.log10(iteraciones + 1)  # Aplicar logaritmo a las iteraciones
log_normas_residuales = np.log10(normas_residuales)  # Aplicar logaritmo a las normas residuales

plt.figure(figsize=(8, 6))
plt.plot(log_iteraciones, log_normas_residuales, marker='o', linestyle='-')
plt.xlabel('Log(Iteración)')
plt.ylabel('Log(Norma Residual)')
plt.title('Norma Residual en función de las Iteraciones (Doble Escala Logarítmica)')
plt.grid(True)

# Guardar la gráfica en un archivo PNG
output_filename = f'{matrix_name}_{solver_name}_norma_residual.png'
plt.savefig(output_filename)

# Guardar información sobre la matriz y el método en un archivo de texto
info_filename = f'{solver_name}_{matrix_name}_info.txt'
with open(info_filename, 'w') as info_file:
    info_file.write(f'Matriz: {matrix_name}\n')
    info_file.write(f'Solver utilizado: {solver_name}\n')
    info_file.write(f'Dimensiones de la matriz A: {filas} x {columnas}\n')
    info_file.write(f'Número de iteraciones: {len(normas_residuales)}\n')

print(f'Los resultados han sido guardados en {output_filename} y {info_filename}')
