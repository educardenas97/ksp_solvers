import scipy.io
from petsc4py import PETSc

# Nombre del archivo .mat
mat_file = 'src/sherman5_2.mat'

# Cargar la matriz .mat
mat_contents = scipy.io.loadmat(mat_file)

# Obtener la matriz A desde el archivo .mat (asegurarse de que sea 'double sparse')
A = mat_contents['A']

# Obtener las dimensiones de la matriz
filas_mat, columnas_mat = A.shape

# Crear la matriz en PETSc (MATSEQAIJ para secuencial o MATMPIAIJ para paralelo)
A_petsc = PETSc.Mat().createAIJ(size=(filas_mat, columnas_mat))

# Utilizar setValues para proporcionar los valores desde el archivo
for fila in range(filas_mat):
    for columna in range(columnas_mat):
        valor = A[fila, columna]
        if valor != 0.0:  # Solo configurar valores no nulos
            A_petsc.setValue(fila, columna, valor)  # Configurar valores usando setValue

A_petsc.assemble()


# Obtener el vector b desde el archivo .mat
b = mat_contents['b'].flatten()  # Asumiendo que 'b' es un vector

# Crear el objeto del solver PETSc
ksp = PETSc.KSP().create()

# Configurar el solver (por ejemplo, usando el método GMRES)
ksp.setType('gmres')

# Configurar opciones adicionales si es necesario
ksp.setFromOptions()

# Crear el vector PETSc para b y configurar su tipo
b_petsc = PETSc.Vec().create()
b_petsc.setSizes(len(b))  # Proporcionar el tamaño como un entero
b_petsc.setType(PETSc.Vec.Type.SEQ)  # O PETSc.Vec.Type.MPI si es un entorno paralelo
b_petsc.setFromOptions()
b_petsc.setArray(b)

# Resolver el sistema Ax=b
x_petsc = PETSc.Vec().create()
x_petsc.setSizes(len(b))
x_petsc.setType(PETSc.Vec.Type.SEQ)
x_petsc.setFromOptions()

# Configurar el solucionador para usar la matriz A_petsc
ksp.setOperators(A_petsc)

# Resolver el sistema Ax=b
ksp.solve(b_petsc, x_petsc)

# Obtener la solución en forma de arreglo de NumPy
x = x_petsc.getArray()

# Imprimir la solución
print("Solución:")
print(x)

# Liberar memoria
A_petsc.destroy()
b_petsc.destroy()
x_petsc.destroy()
ksp.destroy()
