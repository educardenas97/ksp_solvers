import time
import scipy.io
from petsc4py import PETSc
import matplotlib.pyplot as plt
import os
import csv
import sys
import functools
import numpy as np

# Establecer la cantidad deseada de procesos MPI
num_procesos_mpi = 4  # Cambiar este valor al número deseado

# Establecer la variable de entorno PETSC_COMM_WORLD_SIZE
os.environ['PETSC_COMM_WORLD_SIZE'] = str(num_procesos_mpi)


# Decorator to measure execution time
def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.6f} seconds to execute.")
        return result
    return wrapper

# Función para cargar la matriz desde el archivo .mat
def cargar_matriz(mat_file):
    mat_contents = scipy.io.loadmat(mat_file)
    # Ver el contenido del archivo .mat
    # Si tiene el objeto 'Problem', entonces se puede acceder a la matriz A
    # si no, se puede acceder a la matriz A directamente
    if 'Problem' in mat_contents:
        A = mat_contents['Problem']['A'][0][0]
    else:
        A = mat_contents['A']

    filas_mat, columnas_mat = A.shape
    A_petsc = PETSc.Mat().createAIJ(size=(filas_mat, columnas_mat))
    A_petsc.assemblyBegin()
    for fila in range(filas_mat):
        for columna in range(columnas_mat):
            valor = A[fila, columna]
            if valor != 0.0:
                A_petsc.setValue(fila, columna, valor)

    A_petsc.assemblyEnd()
    A_petsc.assemble()
    print("Matriz A cargada")
    print("Dimensiones de la matriz A: ", filas_mat, "x", columnas_mat)
    return A_petsc

# Función para cargar el vector b desde el archivo .mat o definir uno de acuerdo a las dimensiones de A
def cargar_vector_b(mat_file, A):
    mat_contents = scipy.io.loadmat(mat_file)

    if 'Problem' in mat_contents:
        if 'b' in mat_contents:
            b = mat_contents['Problem']['b'][0][0].flatten()
            print("Vector b cargado desde el archivo .mat")
            return b
        else:
            # Define el vector b automáticamente de acuerdo a las dimensiones de A
            filas_A, _ = A.getSize()
            b = np.ones(filas_A)  # Crea un vector de unos con la misma cantidad de filas que A
            print("Vector b definido automáticamente de acuerdo a las dimensiones de A")
    else:
        b = mat_contents['b'].flatten()
        print("Vector b cargado desde el archivo .mat")

    print("Dimensiones del vector b: ", len(b))
    return b
# Función para resolver el sistema con diferentes solvers
def resolver_con_variante(results, solver_type, A_petsc, b):
    print(f"Resolviendo el sistema con {solver_type}...")
    # Crear el objeto del solver PETSc
    ksp = PETSc.KSP().create()

    # Configurar el solver (por ejemplo, usando el método GMRES)
    ksp.setType(solver_type)
    ksp.getPC().setType('none')

    # Configurar opciones adicionales si es necesario
    ksp.setFromOptions()
    ksp.setTolerances(rtol=1e-8, atol=1e-50, max_it=100000)

    # Crear el vector PETSc para b y configurar su tipo
    b_petsc = PETSc.Vec().create()
    b_petsc.setSizes(len(b))
    b_petsc.setType(PETSc.Vec.Type.SEQ)
    b_petsc.setFromOptions()
    b_petsc.setArray(b)

    # Resolver el sistema Ax=b
    x_petsc = PETSc.Vec().create()
    x_petsc.setSizes(len(b))
    x_petsc.setType(PETSc.Vec.Type.SEQ)
    x_petsc.setFromOptions()

    # Configurar el solucionador para usar la matriz A_petsc
    ksp.setOperators(A_petsc)

    # Listas para almacenar las normas residuales y las iteraciones
    normas_residuales = []
    iteraciones = []

    # Función de callback para registrar la norma residual en cada iteración
    def callback(ksp, its, rnorm):
        normas_residuales.append(rnorm)
        iteraciones.append(its)
        results.append([its, rnorm])

    # Configurar la función de callback para registrar las normas residuales
    ksp.setMonitor(callback)

    # Resolver el sistema Ax=b
    ksp.solve(b_petsc, x_petsc)
    num_iteraciones = ksp.getIterationNumber()
    print("Número de ciclos:", num_iteraciones)

    # Obtener la solución en forma de arreglo de NumPy
    x = x_petsc.getArray()
    # print("Información del solver PETSc:")
    # print(ksp.view())
    # print("Solución:")
    # print(x)


# Función para graficar la convergencia
def graficar_convergencia(results, solver_types, mat_file, tiempos_de_resolucion):
    fig, ax1 = plt.subplots()

    for i, (solver_type, result_data) in enumerate(zip(solver_types, results)):
        if not isinstance(result_data, list):
            print(f"Advertencia: No se pudo obtener datos para {solver_type}.")
            continue

        iteraciones = [result[0] for result in result_data]
        normas_residuales = [result[1] for result in result_data]

        # Gráfico para la norma residual en escala logarítmica (eje izquierdo)
        if(solver_type == 'gmres'):
            # Reemplaza el nombre por pdgmres
            ax1.semilogy(iteraciones, normas_residuales, '-o', markersize=2, label=f'pdgmres - {tiempos_de_resolucion[i]:.3f} s')
        else:
            ax1.semilogy(iteraciones, normas_residuales, '-o', markersize=2, label=f'{solver_type} - {tiempos_de_resolucion[i]:.3f} s')

    ax1.set_xlabel('Iteraciones')
    ax1.set_ylabel('Norma Residual (log)')
    ax1.grid(True)
    ax1.legend()

    # Título y leyenda
    matrix_name, _ = os.path.splitext(os.path.basename(mat_file))
    plt.title(f'Convergencia para {matrix_name}')
    plt.tight_layout()

    # Guardar la gráfica como archivo .png
    output_filename = f"{matrix_name}-convergencia.png"
    plt.savefig(output_filename, dpi=600)

    # Mostrar un mensaje cuando se hayan completado todas las ejecuciones
    print("Todas las ejecuciones completadas.")


# Función para graficar los tiempos de resolución
def graficar_tiempos_de_resolucion(tiempos_de_resolucion, solver_types, mat_file):
    fig, ax1 = plt.subplots()

    # Renombrar el solver a "pdgmres" si es "gmres"
    solver_types = ['pdgmres' if solver == 'gmres' else solver for solver in solver_types]

    # Gráfico para los tiempos de resolución (eje izquierdo)
    colors = ['#3894C2' if solver != 'pdgmres' else '#38C2BE' for solver in solver_types]
    
    ax1.bar(solver_types, tiempos_de_resolucion, width=0.5, color=colors)
    ax1.set_xlabel('Métodos', fontweight='bold')
    ax1.set_ylabel('Tiempo de resolución (s)', fontweight='bold')

    # Título y leyenda
    matrix_name, _ = os.path.splitext(os.path.basename(mat_file))
    plt.title(f'Tiempos de resolución para {matrix_name}', fontweight='bold')
    plt.tight_layout()

    # Guardar la gráfica como archivo .png
    output_filename = f"{matrix_name}-tiempos.png"
    plt.savefig(output_filename, dpi=600)

    # Mostrar un mensaje cuando se hayan completado todas las ejecuciones
    print("Todas las ejecuciones completadas.")

# Función para guardar los resultados en un archivo CSV
def guardar_resultados_csv(tiempos_de_resolucion, solver_types, mat_file):
    matrix_name, _ = os.path.splitext(os.path.basename(mat_file))
    output_filename = f"{matrix_name}-resultados.csv"

    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Solver', 'Tiempo de Resolución (s)'])

        for solver, tiempo in zip(solver_types, tiempos_de_resolucion):
            writer.writerow([solver, tiempo])

    print(f"Resultados guardados en {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <mat_file>")
        sys.exit(1)

    mat_file = sys.argv[1]

    # Lista de tipos de solvers a probar
    solver_types = [
        'gmres', 
        'lgmres', 
        'pgmres', 
        'dgmres',
        'pipefgmres', 
        'fgmres'
    ]  # Puedes agregar más según sea necesario

    results = [[] for _ in solver_types]  # Inicializa una lista vacía para cada tipo de solver
    tiempos_de_resolucion = []  # Lista para almacenar los tiempos de resolución

    # Cargar la matriz A una sola vez
    A_petsc = cargar_matriz(mat_file)
    b = cargar_vector_b(mat_file, A_petsc)

    for i, solver_type in enumerate(solver_types):
        # Llamar a la función para resolver el sistema con el tipo de solver actual
        init_time = time.time()
        resolver_con_variante(results[i], solver_type, A_petsc, b)
        excecuting_time = time.time() - init_time
        tiempos_de_resolucion.append(excecuting_time)

    # Generar la gráfica de convergencia con los tiempos de resolución
    graficar_convergencia(results, solver_types, mat_file, tiempos_de_resolucion)

    # Generar gráfica de los tiempos de resolución
    graficar_tiempos_de_resolucion(tiempos_de_resolucion, solver_types, mat_file)

    # Guardar los resultados en un archivo CSV
    guardar_resultados_csv(tiempos_de_resolucion, solver_types, mat_file)

    # Mostrar un mensaje cuando se hayan completado todas las ejecuciones
    print("Todas las ejecuciones completadas.")