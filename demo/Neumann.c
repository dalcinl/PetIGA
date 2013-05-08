#include "petiga.h"

#define pi M_PI

PetscReal Exact(PetscReal x[3])
{
  return sin(pi*x[0]) + sin(pi*x[1]) + sin(pi*x[2]);
}

PetscReal Forcing(PetscReal x[3])
{
  return pi*pi * Exact(x);
}

PetscReal Flux(PetscInt dir,PetscInt side)
{
  return -pi;
}

PETSC_STATIC_INLINE
PetscReal DOT(PetscInt dim,const PetscReal a[],const PetscReal b[])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i]*b[i];
  return s;
}

#undef  __FUNCT__
#define __FUNCT__ "SystemGalerkin"
PetscErrorCode SystemGalerkin(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;

  PetscReal x[3] = {0,0,0};
  IGAPointFormPoint(p,x);
  PetscReal f = Forcing(x);

  const PetscReal *N0,(*N1)[dim];
  IGAPointGetShapeFuns(p,0,(const PetscReal **)&N0);
  IGAPointGetShapeFuns(p,1,(const PetscReal **)&N1);

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++)
      K[a*nen+b] = DOT(dim,N1[a],N1[b]);
    F[a] = N0[a]*f;
  }
  return 0;
}

PETSC_STATIC_INLINE
PetscReal DEL2(PetscInt dim,const PetscReal a[dim][dim])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i][i];
  return s;
}

#undef  __FUNCT__
#define __FUNCT__ "SystemCollocation"
PetscErrorCode SystemCollocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;

  PetscReal x[3] = {0,0,0};
  IGAPointFormPoint(p,x);
  PetscReal f = Forcing(x);

  const PetscReal (*N2)[dim][dim];
  IGAPointGetShapeFuns(p,2,(const PetscReal**)&N2);

  PetscInt a;
  for (a=0; a<nen; a++)
    K[a] += -DEL2(dim,N2[a]);
  F[0] = f;
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "Error"
PetscErrorCode Error(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscReal x[3] = {0,0,0};
  IGAPointFormPoint(p,x);
  PetscScalar u;
  IGAPointFormValue(p,U,&u);
  PetscReal e = PetscAbsScalar(u) -  Exact(x);
  S[0] = e*e;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscBool print_error = PETSC_FALSE;
  PetscBool check_error = PETSC_FALSE;
  PetscBool draw = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Neumann Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-print_error","Prints the L2 error of the solution",__FILE__,print_error,&print_error,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-check_error","Checks the L2 error of the solution",__FILE__,check_error,&check_error,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-draw","If dim <= 2, then draw the solution to the screen",__FILE__,draw,&draw,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  if (iga->dim < 1) {ierr = IGASetDim(iga,2);CHKERRQ(ierr);}
  PetscInt dim,dir,side;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  for (dir=0; dir<dim; dir++) {
    for (side=0; side<2; side++) {
      IGABoundary bnd;
      PetscScalar load = Flux(dir,side);
      ierr = IGAGetBoundary(iga,dir,side,&bnd);CHKERRQ(ierr);
      ierr = IGABoundarySetLoad(bnd,0,load);CHKERRQ(ierr);
    }
  }
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  KSP ksp;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  if (!iga->collocation) {
    ierr = IGASetUserSystem(iga,SystemGalerkin,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  } else {
    ierr = IGASetUserSystem(iga,SystemCollocation,PETSC_NULL);CHKERRQ(ierr);
  }

  {
    MPI_Comm comm;
    MatNullSpace nsp;
    ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(comm,PETSC_TRUE,0,PETSC_NULL,&nsp);CHKERRQ(ierr);
    ierr = KSPSetNullSpace(ksp,nsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);
  }

  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  {
    PetscReal vmin; /* this is a hack */
    ierr = VecMin(x,0,&vmin);CHKERRQ(ierr);
    ierr = VecShift(x,-vmin);CHKERRQ(ierr);
  }

  PetscScalar error;
  ierr = IGAFormScalar(iga,x,1,&error,Error,PETSC_NULL);CHKERRQ(ierr);
  error = PetscSqrtReal(PetscRealPart(error));

  if (print_error) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Error=%G\n",error);CHKERRQ(ierr);}
  if (check_error) {if (error>1e-4) SETERRQ1(PETSC_COMM_WORLD,1,"Error=%G\n",error);}
  if (draw&&dim<3) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
