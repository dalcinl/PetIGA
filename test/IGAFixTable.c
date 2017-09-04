#include "petiga.h"

PETSC_STATIC_INLINE
PetscReal DOT(PetscInt dim,const PetscReal a[],const PetscReal b[])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i]*b[i];
  return s;
}

static PetscReal Solution(PetscInt dim,PetscReal x[])
{
  PetscInt i; PetscReal u = 0.0;
  for (i=0; i<dim; i++) u += x[i]*x[i];
  return u;
}

static PetscReal Forcing(PetscInt dim,PetscReal x[])
{
  PetscInt i; PetscReal f = 0;
  for (i=0; i<dim; i++) f += -2;
  return f;
}

PetscErrorCode System1(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt dim = p->dim;
  PetscInt nen = p->nen;
  PetscReal *N0 = (typeof(N0)) p->shape[0];

  PetscReal x[3];
  IGAPointFormGeomMap(p,x);
  PetscReal g = Solution(dim,x);

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++)
      K[a*nen+b] = N0[a]*N0[b];
    F[a] = N0[a] * g;
  }
  return 0;
}

PetscErrorCode System2(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt dim = p->dim;
  PetscInt nen = p->nen;
  PetscReal *N0        = (typeof(N0)) p->shape[0];
  PetscReal (*N1)[dim] = (typeof(N1)) p->shape[1];

  PetscReal x[3];
  IGAPointFormGeomMap(p,x);
  PetscReal f = Forcing(dim,x);

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++)
      K[a*nen+b] = DOT(dim,N1[a],N1[b]);
    F[a] = N0[a] * f;
  }
  return 0;
}

PetscErrorCode Exact(IGAPoint p,PetscInt order,PetscScalar value[],void *ctx)
{
  PetscReal x[3] = {0,0,0};
  IGAPointFormGeomMap(p,x);
  value[0] = Solution(p->dim,x);
  return 0;
}

int main(int argc, char *argv[])
{
  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscBool print_error = PETSC_FALSE;
  PetscBool check_error = PETSC_FALSE;
  PetscReal error_tol   = 1e-4;
  PetscBool draw = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","IGAFixTable Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-print_error","Prints the L2 error of the solution",__FILE__,print_error,&print_error,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-check_error","Checks the L2 error of the solution",__FILE__,error_tol,&error_tol,&check_error);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-draw","If dim <= 2, then draw the solution to the screen",__FILE__,draw,&draw,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  if (iga->dim < 1) {ierr = IGASetDim(iga,2);CHKERRQ(ierr);}
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);

  /* Solve L2 projection problem */
  ierr = IGASetFormSystem(iga,System1,NULL);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  {
    KSP ksp;
    ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  }

  /* Solve Poisson problem with Dirichlet BCs */
  PetscInt dim,dir,side;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  for (dir=0; dir<dim; dir++)
    for (side=0; side<2; side++)
      {ierr = IGASetBoundaryValue(iga,dir,side,0,/*dummy*/0.0);CHKERRQ(ierr);}
  ierr = IGASetFixTable(iga,x);CHKERRQ(ierr);    /* Set vector to read BCs from */
  ierr = IGASetFormSystem(iga,System2,NULL);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  {
    KSP ksp;
    ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  }
  ierr = IGASetFixTable(iga,NULL);CHKERRQ(ierr); /* Clear vector to read BCs from */

  PetscReal error = 0;
  ierr = IGAComputeErrorNorm(iga,0,x,Exact,&error,NULL);CHKERRQ(ierr);

#if defined(PETSC_USE_REAL_SINGLE)
  error_tol = PetscMax(error_tol,1e-5f);
#endif
  if (print_error) {ierr = PetscPrintf(PETSC_COMM_WORLD,"L2 error = %g\n",(double)error);CHKERRQ(ierr);}
  if (check_error) {if (error>error_tol) SETERRQ2(PETSC_COMM_WORLD,1,"L2 error=%g > %g\n",(double)error,(double)error_tol);}
  if (draw&&dim<3) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
