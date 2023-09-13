# Resolución de Sistemas Lineales con PETSc y Python

Este script de Python, `v3.py`, permite resolver sistemas lineales utilizando la biblioteca PETSc y graficar la convergencia de varios solvers. 



# Instrucciones para Ejecutar el Script

Este script de Python te permite resolver sistemas lineales utilizando diferentes solvers de PETSc y generar gráficos de convergencia. Aquí se proporcionan las instrucciones para ejecutar el script.

## Pasos Previos

Antes de ejecutar el script, asegúrate de tener Python 3.x instalado en tu sistema y sigue estos pasos:

### 1. Crear un Entorno Virtual (Opcional pero recomendado)

Se recomienda crear un entorno virtual para gestionar las dependencias del proyecto. Puedes hacerlo con `virtualenv`:

```bash
# Instalar virtualenv si no está instalado
pip install virtualenv

# Crear un entorno virtual (reemplaza 'myenv' con el nombre que desees)
virtualenv myenv

# Activar el entorno virtual
source myenv/bin/activate
```

### 2. Instalar Dependencias

Dentro del entorno virtual, instala las dependencias necesarias utilizando `pip`:

```bash
pip install -r requirements.txt
```

### 3. Dar Permiso de Ejecución al Script Bash

Asegúrate de que el archivo `main.sh` tenga permisos de ejecución:

```bash
chmod +x main.sh
```

## Ejecutar el Script

Para ejecutar el script y resolver sistemas con diferentes solvers de PETSc, sigue estos pasos:

1. Abre una terminal y asegúrate de estar dentro del entorno virtual (si lo has creado).
2. Ejecuta el script `main.sh` proporcionando el directorio donde se encuentran los archivos .mat como argumento:

```bash
./main.sh /ruta/al/directorio
```

Reemplaza `/ruta/al/directorio` con la ruta real al directorio que contiene los archivos .mat que deseas procesar.

3. El script procesará los archivos .mat dentro del directorio especificado y generará resultados y gráficos de convergencia y tiempos de ejecución en formato .png.

## Dependencias de Python

El script utiliza las siguientes dependencias de Python:

- `petsc4py` (para interactuar con PETSc)
- `scipy` (para cargar archivos .mat)
- `matplotlib` (para generar gráficos)
- `numpy` (para operaciones numéricas)

Puedes instalar estas dependencias utilizando el archivo `requirements.txt`.

