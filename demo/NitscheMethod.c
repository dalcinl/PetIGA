/*

This example solves the Poisson problem using Nitsche's method for
weak imposition of Dirichlet boundary conditions.

@article {NME:NME2863,
author = {Embar, Anand and Dolbow, John and Harari, Isaac},
title = {Imposing Dirichlet boundary conditions with Nitsche's method and spline-based finite elements},
journal = {International Journal for Numerical Methods in Engineering},
volume = {83},
number = {7},
publisher = {John Wiley & Sons, Ltd.},
issn = {1097-0207},
url = {http://dx.doi.org/10.1002/nme.2863},
doi = {10.1002/nme.2863},
pages = {877--898},
keywords = {B-splines, Dirichlet BCs, Nitsche's method, fourth-order problems},
year = {2010},
}

*/

#include "petiga.h"

PETSC_STATIC_INLINE
PetscReal DOT(PetscInt dim,const PetscReal a[],const PetscReal b[])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i]*b[i];
  return s;
}

PETSC_STATIC_INLINE
PetscReal Solution(PetscInt dim,const PetscReal x[])
{
  PetscInt i; PetscReal u = 0.0;
  for (i=0; i<dim; i++) u += x[i]*x[i];
  return u;
}

PETSC_STATIC_INLINE
PetscReal Forcing(PetscInt dim,const PetscReal x[])
{
  PetscInt i; PetscReal f = 0.0;
  for (i=0; i<dim; i++) f += -2.0;
  return f;
}

PETSC_STATIC_INLINE
PetscInt Degree(IGAPoint p)
{
  PetscInt i,dim = p->dim;
  IGAAxis *axis = p->parent->parent->axis;
  PetscInt degree = 0;
  for (i=0; i<dim; i++) degree = PetscMax(degree,axis[i]->p);
  return degree;
}

static PetscReal NormalMeshSize(IGAPoint p)
{
  PetscInt  i,dim = p->dim;
  PetscReal *n = p->normal;
  PetscReal G[dim][dim],N[dim];
  (void)IGAPointFormInvGradGeomMap(p,&G[0][0]);
  for (i=0; i<dim; i++) N[i] = DOT(dim,G[i],n);
  return 2/PetscSqrtReal(DOT(dim,N,N));
}

PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt a,b;

  PetscInt dim = p->dim;
  PetscInt nen = p->nen;
  PetscReal *N0        = (typeof(N0)) p->shape[0];
  PetscReal (*N1)[dim] = (typeof(N1)) p->shape[1];

  PetscReal x[3];
  (void)IGAPointFormGeomMap(p,x);

  if (p->atboundary) goto atboundary;

  PetscReal f = Forcing(dim,x);

  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++)
      K[a*nen+b] = DOT(dim,N1[a],N1[b]);
    F[a] = N0[a] * f;
  }
  return 0;

  atboundary:;

  PetscReal g = Solution(dim,x);
  PetscReal *n = p->normal;

  PetscReal k = Degree(p);
  PetscReal C = 5*(k+1);
  PetscReal h = NormalMeshSize(p);
  PetscReal alpha = C/h;

  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      K[a*nen+b] += - N0[a] * DOT(dim,N1[b],n);
      K[a*nen+b] += - N0[b] * DOT(dim,N1[a],n);
      K[a*nen+b] += + alpha * N0[a]*N0[b];
    }
    F[a] += - DOT(dim,N1[a],n)*g;
    F[a] += + alpha * N0[a]*g;
  }
  return 0;
}

PetscErrorCode Exact(IGAPoint p,PetscInt order,PetscScalar value[],void *ctx)
{
  PetscInt  dim = p->dim;
  PetscReal x[3];
  IGAPointFormGeomMap(p,x);
  value[0] = Solution(dim,x);
  return 0;
}

int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  ierr = IGAOptionsAlias("-d",NULL,"-iga_dim");
  ierr = IGAOptionsAlias("-N",NULL,"-iga_elements");
  ierr = IGAOptionsAlias("-L",NULL,"-iga_limits");
  ierr = IGAOptionsAlias("-p",NULL,"-iga_degree");
  ierr = IGAOptionsAlias("-k",NULL,"-iga_continuity");
  ierr = IGAOptionsAlias("-q",NULL,"-iga_quadrature");

  PetscBool print_error = PETSC_FALSE;
  PetscBool check_error = PETSC_FALSE;
  PetscReal error_tol   = 1e-4;
  PetscBool draw = PETSC_FALSE;
  PetscBool save = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","NitscheMethod Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-print_error","Prints the L2 error of the solution",__FILE__,print_error,&print_error,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-check_error","Checks the L2 error of the solution",__FILE__,error_tol,&error_tol,&check_error);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-draw","If dim <= 2, then draw the solution to the screen",__FILE__,draw,&draw,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-save","Save the solution to file",                        __FILE__,save,&save,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  if (iga->dim < 1) {ierr = IGASetDim(iga,2);CHKERRQ(ierr);}
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  PetscInt dim,axis,side;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  for (axis=0; axis<dim; axis++)
    for (side=0; side<2; side++)
      {ierr = IGASetBoundaryForm(iga,axis,side,PETSC_TRUE);CHKERRQ(ierr);}

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetFormSystem(iga,System,NULL);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  PetscReal error;
  ierr = IGAComputeErrorNorm(iga,0,x,Exact,&error,NULL);CHKERRQ(ierr);

  if (print_error) {ierr = PetscPrintf(PETSC_COMM_WORLD,"L2 error = %g\n",(double)error);CHKERRQ(ierr);}
  if (check_error) {if (error>error_tol) SETERRQ2(PETSC_COMM_WORLD,1,"L2 error=%g > %g\n",(double)error,(double)error_tol);}
  if (draw&&dim<3) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  if (save)        {ierr = IGAWrite(iga,"NitscheGeometry.dat");CHKERRQ(ierr);}
  if (save)        {ierr = IGAWriteVec(iga,x,"NitscheSolution.dat");CHKERRQ(ierr);}

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
