import time
import scipy.io
from petsc4py import PETSc
import matplotlib.pyplot as plt
import os
import csv
import sys
import functools

# Establecer la cantidad deseada de procesos MPI
num_procesos_mpi = 4  # Cambiar este valor al número deseado

# Establecer la variable de entorno PETSC_COMM_WORLD_SIZE
os.environ['PETSC_COMM_WORLD_SIZE'] = str(num_procesos_mpi)

# resultados
results = []


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
@measure_time
def cargar_matriz(mat_file):
    mat_contents = scipy.io.loadmat(mat_file)
    A = mat_contents['A']
    filas_mat, columnas_mat = A.shape
    A_petsc = PETSc.Mat().createAIJ(size=(filas_mat, columnas_mat))
    A_petsc.assemblyBegin() # Assembling the matrix makes it "useable".
    for fila in range(filas_mat):
        for columna in range(columnas_mat):
            valor = A[fila, columna]
            if valor != 0.0:
                A_petsc.setValue(fila, columna, valor)

    A_petsc.assemblyEnd()
    A_petsc.assemble()
    print("Matriz A cargada")
    # Mostrar las dimensiones de la matriz
    print("Dimensiones de la matriz A: ", filas_mat, "x", columnas_mat)
    
    return A_petsc

# Función para cargar el vector b desde el archivo .mat
@measure_time
def cargar_vector_b(mat_file):
    mat_contents = scipy.io.loadmat(mat_file)
    b = mat_contents['b'].flatten()
    print("Vector b cargado")

    # Mostrar las dimensiones del vector
    print("Dimensiones del vector b: ", len(b))
    return b

# Función principal
@measure_time
def resolver_sistema(mat_file):
    # Cargar la matriz A
     # Medir el tiempo de resolución
    print("Cargando matriz A...")
    A_petsc = cargar_matriz(mat_file)
    print("Matriz A cargada")
    # Cargar el vector b
    print("Cargando vector b...")
    b = cargar_vector_b(mat_file)
    print("Vector b cargado")
    # Crear el objeto del solver PETSc
    ksp = PETSc.KSP().create()

    # Configurar el solver (por ejemplo, usando el método GMRES)
    ksp.setType(solver_type)
    # Configurar que no se utilice precondicionador
    ksp.getPC().setType('none')
    def prueba_convergencia_sin_precondicionamiento(ksp, its, rnorm):
        
        return its >= 1000  # Reemplaza el 2 con el índice apropiado para max_it

    # ksp.setConvergenceTest(prueba_convergencia_sin_precondicionamiento)
    # Configurar opciones adicionales si es necesario
    ksp.setFromOptions()
    ksp.setTolerances(rtol=1e-8, atol=1e-50, max_it=200000)
    ksp.getPC().setType('none')
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

    print("Resolviendo el sistema...")
    # Resolver el sistema Ax=b
    init_time = time.time()
    ksp.solve(b_petsc, x_petsc)
    excecuting_time = time.time() - init_time
    print("Tiempo de resolución:", excecuting_time, "segundos")
    num_iteraciones = ksp.getIterationNumber()
    print("Número de ciclos:", num_iteraciones)
    
    # Obtener la solución en forma de arreglo de NumPy
    x = x_petsc.getArray()
    # Mostrar la información del proceso de resolución
    print("Información del solver PETSc:")
    print(ksp.view())
    # Imprimir la solución
    print("Solución:")
    print(x)
    # Crear el gráfico de la norma residual en función de los ciclos
    matrix_name, _ = os.path.splitext(os.path.basename(mat_file))

    plt.figure()
    plt.semilogy(iteraciones, normas_residuales, '-o', markersize=2)
    plt.xlabel('Iteraciones')
    plt.ylabel('Norma Residual (log)')
    plt.title(f'Convergencia {matrix_name} - {solver_type} - {excecuting_time:.6f} segundos')
    plt.grid(True)
    
    # Guardar el gráfico como un archivo .png
    output_filename = f"{matrix_name}-{solver_type}.png"
    plt.savefig(output_filename)
    
    x = x_petsc.getArray()
    
    # Imprimir la solución
    print("Solución:")
    print(x)
    
    try:
        # Liberar memoria
        A_petsc.destroy()
        b_petsc.destroy()
        x_petsc.destroy()
        ksp.destroy()
    except:
        print("Error al liberar memoria")
        pass

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <mat_file> <solver_type> <num_procesos_mpi>")
        sys.exit(1)

    mat_file = sys.argv[1]
    solver_type = sys.argv[2]
    num_procesos_mpi = int(sys.argv[3])
    resolver_sistema(mat_file)
    
    matrix_name, _ = os.path.splitext(os.path.basename(mat_file))
    output_filename = f"{matrix_name}-{solver_type}.csv"
    # Save the results to a CSV file
    with open(output_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['iteracion', 'norma_residual'])  # Write header
        csv_writer.writerows(results)  # Write the results