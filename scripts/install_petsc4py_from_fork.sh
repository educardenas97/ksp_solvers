#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${PETSC_DIR:-}" ]]; then
  echo "Error: define PETSC_DIR apuntando al root de tu fork de PETSc."
  echo "Ejemplo: export PETSC_DIR=$HOME/dev/petsc"
  exit 1
fi

if [[ ! -d "$PETSC_DIR" ]]; then
  echo "Error: PETSC_DIR no existe: $PETSC_DIR"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CYTHON_VERSION="${CYTHON_VERSION:-3.0.11}"
PETSC4PY_LOCAL_DIR="${PETSC4PY_LOCAL_DIR:-}"

PETSC_VERSION_MAJOR="$(grep -E '^#define PETSC_VERSION_MAJOR' "$PETSC_DIR/include/petscversion.h" | awk '{print $3}')"
PETSC_VERSION_MINOR="$(grep -E '^#define PETSC_VERSION_MINOR' "$PETSC_DIR/include/petscversion.h" | awk '{print $3}')"

PETSC4PY_VERSION="${PETSC4PY_VERSION:-${PETSC_VERSION_MAJOR}.${PETSC_VERSION_MINOR}.*}"

PETSC_ARCH_VALUE="${PETSC_ARCH:-}"
if [[ -n "$PETSC_ARCH_VALUE" && ! -d "$PETSC_DIR/$PETSC_ARCH_VALUE" ]]; then
  echo "Aviso: PETSC_ARCH=$PETSC_ARCH_VALUE no existe en $PETSC_DIR; se instalará sin PETSC_ARCH explícito."
  PETSC_ARCH_VALUE=""
fi

echo "Usando Python: $PYTHON_BIN"
echo "Usando PETSC_DIR: $PETSC_DIR"
echo "Usando PETSC_ARCH: ${PETSC_ARCH_VALUE:-<no definido>}"
if [[ -n "$PETSC4PY_LOCAL_DIR" ]]; then
  echo "Instalando petsc4py desde fork local: $PETSC4PY_LOCAL_DIR"
else
  echo "Instalando petsc4py==${PETSC4PY_VERSION} desde source..."
fi
echo "Usando Cython==$CYTHON_VERSION"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install --upgrade "cython==$CYTHON_VERSION"
"$PYTHON_BIN" -m pip install --upgrade "numpy>=1.26,<2"

if [[ -n "$PETSC4PY_LOCAL_DIR" ]]; then
  if [[ ! -d "$PETSC4PY_LOCAL_DIR" ]]; then
    echo "Error: PETSC4PY_LOCAL_DIR no existe: $PETSC4PY_LOCAL_DIR"
    exit 1
  fi

  if [[ -f "$PETSC4PY_LOCAL_DIR/src/binding/petsc4py/pyproject.toml" ]]; then
    PETSC4PY_INSTALL_TARGET="$PETSC4PY_LOCAL_DIR/src/binding/petsc4py"
  elif [[ -f "$PETSC4PY_LOCAL_DIR/pyproject.toml" ]]; then
    PETSC4PY_INSTALL_TARGET="$PETSC4PY_LOCAL_DIR"
  else
    echo "Error: no se encontró pyproject.toml en PETSC4PY_LOCAL_DIR ni en src/binding/petsc4py"
    exit 1
  fi
fi

if [[ -n "$PETSC_ARCH_VALUE" ]]; then
  if [[ -n "$PETSC4PY_LOCAL_DIR" ]]; then
    PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH_VALUE" "$PYTHON_BIN" -m pip install --no-build-isolation --no-binary=:all: "$PETSC4PY_INSTALL_TARGET"
  else
    PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH_VALUE" "$PYTHON_BIN" -m pip install --no-build-isolation --no-binary=petsc4py "petsc4py==$PETSC4PY_VERSION"
  fi
else
  if [[ -n "$PETSC4PY_LOCAL_DIR" ]]; then
    env -u PETSC_ARCH PETSC_DIR="$PETSC_DIR" "$PYTHON_BIN" -m pip install --no-build-isolation --no-binary=:all: "$PETSC4PY_INSTALL_TARGET"
  else
    env -u PETSC_ARCH PETSC_DIR="$PETSC_DIR" "$PYTHON_BIN" -m pip install --no-build-isolation --no-binary=petsc4py "petsc4py==$PETSC4PY_VERSION"
  fi
fi

echo "OK: petsc4py instalado contra PETSc custom."
