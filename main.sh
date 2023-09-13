#!/bin/bash

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
python_script="src/petscPy/v3.py"

# Loop a través de los archivos .mat dentro del directorio y ejecutar el script Python
for mat_file in "./$directory_path"/*.mat; do
    if [ -f "$mat_file" ]; then
        echo "Procesando $mat_file..."
        python3 "$python_script" "$mat_file"
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "Error al procesar $mat_file (Código de salida: $exit_code)"
        fi
    fi
done

echo "Proceso completado."
