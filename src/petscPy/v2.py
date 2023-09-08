import time
import scipy.io
from petsc4py import PETSc
import matplotlib.pyplot as plt

# Función para cargar la matriz desde el archivo .mat
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
                print(fila,",", columna, ":", valor)

    A_petsc.assemblyEnd()

    A_petsc.assemble()
    print("Matriz A cargada")
    # Mostrar la dimensiones de la matriz
    print("Dimensiones de la matriz A: ", filas_mat, "x", columnas_mat)
    
    return A_petsc

# Función para cargar el vector b desde el archivo .mat
def cargar_vector_b(mat_file):
    mat_contents = scipy.io.loadmat(mat_file)
    b = mat_contents['b'].flatten()
    print("Vector b cargado")

    # Mostrar la dimensiones del vector
    print("Dimensiones del vector b: ", len(b))
    return b

# Función principal
def resolver_sistema(mat_file):
    # Cargar la matriz A
     # Medir el tiempo de resolución
    start_time = time.time()
    print("Cargando matriz A...")
    A_petsc = cargar_matriz(mat_file)
    print("Matriz A cargada")
    # Cargar el vector b
    print("Cargando vector b...")
    b = cargar_vector_b(mat_file)
    print("Vector b cargado")
    end_time = time.time()
    print("Tiempo de carga:", end_time - start_time, "segundos")
    # Crear el objeto del solver PETSc
    ksp = PETSc.KSP().create()
    
    # Configurar el solver (por ejemplo, usando el método GMRES)
    ksp.setType('lgmres')
    
    # Configurar opciones adicionales si es necesario
    ksp.setFromOptions()
    ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=1000000)
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
    
    # Configurar la función de callback para registrar las normas residuales
    ksp.setMonitor(callback)

    # Medir el tiempo de resolución
    start_time = time.time()
    print("Resolviendo el sistema...")
    # Resolver el sistema Ax=b
    ksp.solve(b_petsc, x_petsc)
    
    end_time = time.time()
    print("Tiempo de resolución:", end_time - start_time, "segundos")
    
    # Obtener la solución en forma de arreglo de NumPy
    x = x_petsc.getArray()

    # Mostrar la información del proceso de resolución
    print("Información del solver PETSc:")
    print(ksp.view())

    
    # Imprimir la solución
    print("Solución:")
    print(x)
    
      # Crear el gráfico de la norma residual en función de las iteraciones
    plt.figure()
    plt.semilogy(iteraciones, normas_residuales, '-o', markersize=2)
    plt.xlabel('Iteraciones')
    plt.ylabel('Norma Residual (log)')
    plt.title('Convergencia del Solver PETSc')
    plt.grid(True)
    
    # Guardar el gráfico como un archivo .png
    plt.savefig('lgmres.png')
    
    x = x_petsc.getArray()
    
    # Imprimir la solución
    print("Solución:")
    print(x)
    
    # Liberar memoria
    A_petsc.destroy()
    b_petsc.destroy()
    x_petsc.destroy()
    ksp.destroy()

if __name__ == "__main__":
    mat_file = 'src/sherman5_2.mat'
    resolver_sistema(mat_file)
