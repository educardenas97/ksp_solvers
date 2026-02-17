# PDGMRES — GMRES con Reinicio Adaptativo por Controlador Proporcional-Derivativo

Solver iterativo para sistemas lineales $Ax = b$ basado en GMRES con reinicio adaptativo.
El parámetro de reinicio $m$ se ajusta automáticamente entre ciclos mediante un controlador PD que monitorea la tasa de convergencia.

> **Referencia principal:**
> Cuevas, Schaerer, Bhaya — *"A proportional-derivative control strategy for restarting the GMRES(m) algorithm"*, J. Comp. Appl. Math. 337:209-224, 2018.
> Cabral, Schaerer, Bhaya — *"Improving GMRES(m) using an adaptive switching controller"*, Numer. Linear Algebra Appl. 27(5):e2305, 2020.

---

## Tabla de contenidos

1. [Descripción del algoritmo](#descripción-del-algoritmo)
2. [Parámetros ajustables](#parámetros-ajustables)
3. [Guía de instalación](#guía-de-instalación)
4. [Uso con Python (petsc4py)](#uso-con-python-petsc4py)
5. [Ejecución del benchmark](#ejecución-del-benchmark)
6. [Estructura del proyecto](#estructura-del-proyecto)

---

## Descripción del algoritmo

PDGMRES extiende GMRES(m) reemplazando el reinicio fijo por una ley de control PD:

$$
m_{j+1} = m_j + \left\lfloor \alpha_p \cdot \frac{\|r_j\|}{\|r_{j-1}\|} + \alpha_d \cdot \frac{\|r_j\| - \|r_{j-2}\|}{2\,\|r_{j-1}\|} \right\rfloor
$$

| Término | Significado |
|---------|-------------|
| $\alpha_p \cdot \text{ratio}$ | **Proporcional** — reacciona a la tasa de reducción del residuo actual |
| $\alpha_d \cdot \text{deriv}$ | **Derivativo** — anticipa la tendencia usando el historial de dos ciclos |

Cuando $m$ cae por debajo de `m_min` (estancamiento detectado), se incrementa un contador de estancamiento y se reinicia $m$ a:

$$
m_{\text{new}} = m_{\text{init}} + \text{stag\_count} \cdot m_{\text{step}}
$$

Esto permite al solver escapar de ciclos improductivos sin intervención manual.

---

## Parámetros ajustables

### Parámetros del controlador PD

| Parámetro | Valor por defecto | Descripción |
|-----------|:-----------------:|-------------|
| `alpha_p` | `-0.625` | Ganancia proporcional. Controla la reacción ante la tasa de convergencia actual. |
| `alpha_d` | `4.375` | Ganancia derivativa. Controla la anticipación basada en la tendencia del residuo. |

### Parámetros de reinicio

| Parámetro | Valor por defecto | Descripción |
|-----------|:-----------------:|-------------|
| `m_init` | `10` | Tamaño inicial del espacio de Krylov (reinicio inicial). |
| `m_min` | `3` | Tamaño mínimo permitido de reinicio. Si $m$ baja de este valor, se detecta estancamiento. |
| `m_step` | `10` | Incremento aplicado al escalón de escape ante estancamiento. |
| `m_max` | `60` | Tamaño máximo de reinicio (techo de pre-asignación de memoria). Equivale a `m_init + 5 * m_step`. |

### Parámetros heredados de GMRES

| Opción de línea de comandos | Descripción |
|------------------------------|-------------|
| `-ksp_gmres_restart <n>` | Establece el tamaño *inicial* de reinicio (equivale a `m_init`). |
| `-ksp_gmres_haptol <tol>` | Tolerancia para detección de "happy ending" (convergencia exacta). Default: `1e-30`. |
| `-ksp_gmres_preallocate` | Pre-asigna todos los vectores de Krylov desde el inicio. |
| `-ksp_gmres_classicalgramschmidt` | Usa Gram-Schmidt clásico (default). |
| `-ksp_gmres_modifiedgramschmidt` | Usa Gram-Schmidt modificado (más estable, más lento). |
| `-ksp_gmres_cgs_refinement_type <tipo>` | Tipo de refinamiento CGS: `refine_never`, `refine_ifneeded`, `refine_always`. |

### Parámetros generales del solver KSP

| Opción | Descripción |
|--------|-------------|
| `-ksp_rtol <tol>` | Tolerancia relativa de convergencia. |
| `-ksp_atol <tol>` | Tolerancia absoluta de convergencia. |
| `-ksp_max_it <n>` | Número máximo de iteraciones. |
| `-pc_type <tipo>` | Tipo de precondicionador (`none`, `jacobi`, `ilu`, `lu`, etc.). |

### Precondicionamiento soportado

| Norma | Lado del PC | Prioridad |
|-------|-------------|:---------:|
| `KSP_NORM_PRECONDITIONED` | `PC_LEFT` | 4 (máxima) |
| `KSP_NORM_UNPRECONDITIONED` | `PC_RIGHT` | 3 |
| `KSP_NORM_PRECONDITIONED` | `PC_SYMMETRIC` | 2 |
| `KSP_NORM_NONE` | `PC_RIGHT` | 1 |
| `KSP_NORM_NONE` | `PC_LEFT` | 1 |

---

## Guía de instalación

### Requisitos previos

- **Python** >= 3.9
- **GCC / Clang** con soporte C99
- **Make** y **CMake**
- **Git**
- **MPI** (OpenMPI o MPICH)

### 1. Clonar el fork de PETSc con PDGMRES

```bash
git clone https://github.com/educardenas97/ksp_solvers.git
cd ksp_solvers
```

El fork de PETSc se encuentra como directorio adyacente. La estructura esperada es:

```
nidtec-fpuna/
├── ksp_solvers/    ← scripts de benchmark y main.py
└── petsc/          ← fork de PETSc con KSPPDGMRES
```

Si el fork de PETSc está en un repositorio separado, clónalo:

```bash
git clone https://gitlab.com/eduardocardenas97/petsc.git ../petsc
```

### 2. Compilar PETSc

```bash
cd ../petsc

# Configurar (ajustar según tu sistema)
./configure --with-debugging=1 \
            --download-mpich \
            --download-fblaslapack

# Compilar
make all
make check   # opcional, ejecuta tests básicos
```

### 3. Crear entorno virtual de Python

```bash
cd ../ksp_solvers

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Instalar petsc4py desde el fork

El script `scripts/install_petsc4py_from_fork.sh` compila `petsc4py` contra tu build local de PETSc.  
Fija una versión compatible de Cython, instala NumPy, y realiza la instalación con `--no-build-isolation` para evitar errores de compilación en Python >= 3.12.

```bash
export PETSC_DIR=$(realpath ../petsc)
export PETSC_ARCH=arch-linux-c-debug    # ajustar según tu configuración

chmod +x scripts/install_petsc4py_from_fork.sh
./scripts/install_petsc4py_from_fork.sh
```

#### Variables de entorno opcionales

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `PETSC_DIR` | **Requerido.** Ruta al directorio raíz del fork de PETSc. | `/home/user/petsc` |
| `PETSC_ARCH` | Arquitectura de compilación de PETSc. | `arch-linux-c-debug` |
| `PETSC4PY_LOCAL_DIR` | Ruta local al fork de petsc4py. Acepta la raíz del repo PETSc (detecta `src/binding/petsc4py` automáticamente) o la raíz de un repo petsc4py independiente. | `/home/user/petsc` |
| `PETSC4PY_VERSION` | Versión específica de petsc4py a instalar. Por defecto se infiere de la versión `major.minor` de PETSc (e.g., PETSc 3.24.x → petsc4py 3.24.*). | `3.24.5` |
| `CYTHON_VERSION` | Versión de Cython a utilizar. | `3.0.11` |
| `PYTHON_BIN` | Intérprete Python a utilizar. | `./.venv/bin/python` |

#### Ejemplo con fork local de petsc4py

```bash
export PETSC_DIR=/ruta/a/tu/fork/petsc
export PETSC_ARCH=arch-linux-c-debug

PETSC4PY_LOCAL_DIR=/ruta/a/tu/fork/petsc \
PYTHON_BIN=./.venv/bin/python \
./scripts/install_petsc4py_from_fork.sh
```

> **Nota:** la compilación de `petsc4py` desde source puede tardar varios minutos. Si se interrumpe con `Ctrl+C`, la instalación quedará incompleta.

### 5. Verificar la instalación

```bash
source .venv/bin/activate
python -c "
from petsc4py import PETSc
ksp = PETSc.KSP().create()
ksp.setType('pdgmres')
print('PDGMRES type:', ksp.getType())
print('Instalación exitosa.')
"
```

---

## Uso con Python (petsc4py)

### Ejemplo básico

```python
from petsc4py import PETSc
import numpy as np

# Crear una matriz de ejemplo (5x5 diagonal dominante)
n = 5
A = PETSc.Mat().createAIJ(size=(n, n))
A.setUp()
for i in range(n):
    A.setValue(i, i, 4.0)
    if i > 0:
        A.setValue(i, i - 1, -1.0)
    if i < n - 1:
        A.setValue(i, i + 1, -1.0)
A.assemblyBegin()
A.assemblyEnd()

# Crear vector RHS
b = PETSc.Vec().createSeq(n)
b.set(1.0)

# Crear vector solución
x = PETSc.Vec().createSeq(n)

# Configurar el solver PDGMRES
ksp = PETSc.KSP().create()
ksp.setType("pdgmres")
ksp.setOperators(A)
ksp.getPC().setType("none")
ksp.setTolerances(rtol=1e-8, atol=1e-8, max_it=10000)

# Resolver
ksp.solve(b, x)

print(f"Iteraciones: {ksp.getIterationNumber()}")
print(f"Norma residual: {ksp.getResidualNorm():.2e}")
print(f"Solución: {x.getArray()}")
```

### Uso con el enum PETSc.KSP.Type

```python
from petsc4py import PETSc

ksp = PETSc.KSP().create()
ksp.setType(PETSc.KSP.Type.PDGMRES)
```

### Configurar parámetros por línea de comandos

Los parámetros del controlador PD y de GMRES se pueden pasar como opciones al ejecutar el script:

```bash
python src/main.py data/cavity05.mat \
    -ksp_type pdgmres \
    -ksp_gmres_restart 15 \
    -ksp_rtol 1e-10 \
    -ksp_monitor
```

### Monitoreo de convergencia con callback

```python
def monitor(ksp, its, rnorm):
    print(f"  Iteración {its}: ||r|| = {rnorm:.6e}")

ksp = PETSc.KSP().create()
ksp.setType("pdgmres")
ksp.setMonitor(monitor)
ksp.setOperators(A)
ksp.solve(b, x)
```

### Ejemplo: cargar matriz desde archivo .mat

```python
import scipy.io
from petsc4py import PETSc

# Cargar desde archivo SuiteSparse/MATLAB
mat = scipy.io.loadmat("data/cavity05.mat")
A_scipy = mat["Problem"]["A"][0][0] if "Problem" in mat else mat["A"]

rows, cols = A_scipy.shape
A = PETSc.Mat().createAIJ(size=(rows, cols))
A.setUp()
for i in range(rows):
    for j in range(cols):
        v = A_scipy[i, j]
        if v != 0.0:
            A.setValue(i, j, v)
A.assemblyBegin()
A.assemblyEnd()

b = PETSc.Vec().createSeq(rows)
b.set(1.0)
x = PETSc.Vec().createSeq(rows)

ksp = PETSc.KSP().create()
ksp.setType("pdgmres")
ksp.setOperators(A)
ksp.getPC().setType("none")
ksp.setTolerances(rtol=1e-8, atol=1e-8, max_it=100000)
ksp.solve(b, x)

print(f"Convergió en {ksp.getIterationNumber()} iteraciones")
```

---

## Ejecución del benchmark

El proyecto incluye un script de benchmark que compara PDGMRES contra otras variantes de GMRES.

### Solvers incluidos en la comparación

| Solver | Tipo PETSc | Descripción |
|--------|------------|-------------|
| **pdgmres** | `KSPPDGMRES` | GMRES con reinicio adaptativo PD *(este fork)* |
| gmres | `KSPGMRES` | GMRES estándar con reinicio fijo |
| fgmres | `KSPFGMRES` | GMRES flexible |
| lgmres | `KSPLGMRES` | GMRES con reinicio acelerado |
| dgmres | `KSPDGMRES` | GMRES deflacionado |
| pgmres | `KSPPGMRES` | GMRES pipelined |
| pipefgmres | `KSPPIPEFGMRES` | GMRES flexible pipelined |

### Personalización de solvers

Para modificar la lista de solvers a comparar, editar el arreglo `solver_types` en `src/main.py`:

```python
solver_types = [
    "pdgmres",
    "lgmres",
    "pgmres",
    "dgmres",
    "pipefgmres",
    "fgmres",
    "gmres",
]
```

Las tolerancias y otros parámetros se ajustan en el método `resolver_con_variante` de `src/main.py`:

```python
ksp.setTolerances(rtol=1e-8, atol=1e-8, max_it=100000)
```

### Ejecutar con una sola matriz

```bash
source .venv/bin/activate
python src/main.py data/cavity05.mat
```

### Ejecutar con todas las matrices del directorio

```bash
chmod +x main.sh
./main.sh data/
```

### Salidas generadas

Para cada matriz `<nombre>.mat` se generan:

| Archivo | Contenido |
|---------|-----------|
| `<nombre>-convergencia.png` | Gráfico de norma residual vs. iteraciones (escala log) |
| `<nombre>-tiempos.png` | Gráfico de barras con tiempos de resolución por solver |
| `<nombre>-resultados.csv` | CSV con tiempos de resolución por solver |

---

## Dependencias de Python

| Paquete | Uso |
|---------|-----|
| `petsc4py` | Interfaz Python para PETSc (se instala desde el fork) |
| `scipy` | Carga de archivos `.mat` |
| `matplotlib` | Generación de gráficos |
| `numpy` | Operaciones numéricas |

Las dependencias (excepto petsc4py) se instalan con:

```bash
pip install -r requirements.txt
```

---

## Estructura del proyecto

```
ksp_solvers/
├── readme.md                          ← este archivo
├── requirements.txt                   ← dependencias Python (sin petsc4py)
├── main.sh                            ← script bash para ejecución por lotes
├── scripts/
│   └── install_petsc4py_from_fork.sh  ← instalador de petsc4py desde fork
├── src/
│   └── main.py                        ← script principal de benchmark
└── data/
    ├── cavity05.mat                   ← matrices de prueba (formato .mat)
    ├── cavity10.mat
    ├── ...
    └── young3c.mat
```

### Archivos clave del fork de PETSc

```
petsc/
├── src/ksp/ksp/impls/gmres/pdgmres/
│   ├── pdgmres.c              ← implementación del solver PDGMRES
│   ├── pdgmresimpl.h          ← estructura KSP_PDGMRES y macros
│   └── makefile
├── include/petscksp.h          ← define KSPPDGMRES = "pdgmres"
└── src/binding/petsc4py/
    ├── src/petsc4py/PETSc/
    │   ├── KSP.pyx            ← binding Python (Type.PDGMRES)
    │   └── petscksp.pxi       ← declaración Cython de KSPPDGMRES
    └── test/test_ksp.py        ← test unitario TestKSPPDGMRES
```

---

## Licencia

PETSc se distribuye bajo licencia BSD 2-Clause. Consultar el archivo `LICENSE` en el directorio `petsc/`.