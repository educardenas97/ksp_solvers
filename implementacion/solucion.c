#include <petsc.h>

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);

    Mat A;
    Vec b;
    PetscViewer viewer;
    KSP solver;
    Vec x;

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, "archivo.mat", FILE_MODE_READ, &viewer);
    MatCreate(PETSC_COMM_WORLD, &A);
    MatLoad(A, viewer);
    PetscViewerDestroy(&viewer);

    VecCreate(PETSC_COMM_WORLD, &b);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, "vector_b.mat", FILE_MODE_READ, &viewer);
    VecLoad(b, viewer);
    PetscViewerDestroy(&viewer);

    KSPCreate(PETSC_COMM_WORLD, &solver);
    KSPSetType(solver, KSPGMRES);

    KSPSetOperators(solver, A, A);

    VecCreate(PETSC_COMM_WORLD, &x);
    KSPSolve(solver, b, x);

    KSPDestroy(&solver);
    VecDestroy(&x);
    MatDestroy(&A);
    VecDestroy(&b);

    PetscFinalize();
    return 0;
}
