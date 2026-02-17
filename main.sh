#!/bin/bash

# Configuración de PETSc (necesario para petsc4py con fork local)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PETSC_DIR="${PETSC_DIR:-$(realpath "$SCRIPT_DIR/../petsc")}"
export PETSC_ARCH="${PETSC_ARCH:-arch-linux-c-debug}"

# Verificar si se proporcionó un directorio
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

directory_path="$1"

# Verificar si el directorio existe
if [ ! -d "$directory_path" ]; then
    echo "El directorio $directory_path no existe."
    exit 1
fi

# Ruta al script Python
python_script="src/main.py"

if [ -n "${PYTHON_BIN:-}" ]; then
    python_cmd="$PYTHON_BIN"
elif [ -x "./.venv/bin/python" ]; then
    python_cmd="./.venv/bin/python"
else
    python_cmd="python3"
fi

echo "Usando intérprete: $python_cmd"
echo "Usando PETSC_DIR: $PETSC_DIR"
echo "Usando PETSC_ARCH: $PETSC_ARCH"

# Loop a través de los archivos .mat dentro del directorio y ejecutar el script Python
for mat_file in "./$directory_path"/*.mat; do
    if [ -f "$mat_file" ]; then
        echo "Procesando $mat_file..."
        "$python_cmd" "$python_script" "$mat_file"
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "Error al procesar $mat_file (Código de salida: $exit_code)"
        fi
    fi
done

echo "Proceso completado."
