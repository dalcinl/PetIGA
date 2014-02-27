#include "petiga.h"

#if PETSC_VERSION_LT(3,5,0)
#define KSPSetOperators(ksp,A,B) KSPSetOperators(ksp,A,B,SAME_NONZERO_PATTERN)
#endif

#define pi M_PI

PetscReal Exact(PetscReal x[3])
{
  return sin(2*pi*x[0]) + sin(2*pi*x[1]) + sin(2*pi*x[2]);
}

PetscReal Forcing(PetscReal x[3])
{
  return 4*pi*pi * Exact(x);
}

PetscReal Flux(PetscInt dir,PetscInt side)
{
  return (side?+1:-1) * 2*pi;
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
  const PetscReal *N0        = (typeof(N0)) p->shape[0];
  const PetscReal (*N1)[dim] = (typeof(N1)) p->shape[1];

  PetscReal x[3] = {0,0,0};
  IGAPointFormPoint(p,x);
  PetscReal f = Forcing(x);

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
  const PetscReal (*N2)[dim][dim] = (typeof(N2)) p->shape[2];

  PetscReal x[3] = {0,0,0};
  IGAPointFormPoint(p,x);
  PetscReal f = Forcing(x);

  PetscInt a;
  for (a=0; a<nen; a++)
    K[a] += -DEL2(dim,N2[a]);
  F[0] = f;
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "MassVector"
PetscErrorCode MassVector(IGAPoint p,PetscScalar *V,void *ctx)
{
  PetscInt a,nen = p->nen;
  const PetscReal *N0 = (typeof(N0)) p->shape[0];
  for (a=0; a<nen; a++) V[a] = N0[a];
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
  PetscScalar e = u -  Exact(x);
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
  ierr = PetscOptionsBool("-print_error","Prints the L2 error of the solution",__FILE__,print_error,&print_error,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-check_error","Checks the L2 error of the solution",__FILE__,check_error,&check_error,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-draw","If dim <= 2, then draw the solution to the screen",__FILE__,draw,&draw,NULL);CHKERRQ(ierr);
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
      PetscScalar load = Flux(dir,side);
      ierr = IGASetBoundaryLoad(iga,dir,side,0,load);CHKERRQ(ierr);
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
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  if (!iga->collocation) {
    ierr = IGASetFormSystem(iga,SystemGalerkin,NULL);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  } else {
    ierr = IGASetFormSystem(iga,SystemCollocation,NULL);CHKERRQ(ierr);
  }

  {
    MPI_Comm comm;
    MatNullSpace nsp;
    ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(comm,PETSC_TRUE,0,NULL,&nsp);CHKERRQ(ierr);
    ierr = KSPSetNullSpace(ksp,nsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);
  }

  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  Vec Q;
  ierr = IGACreateVec(iga,&Q);CHKERRQ(ierr);
  ierr = IGASetFormVector(iga,MassVector,NULL);CHKERRQ(ierr);
  ierr = IGAComputeVector(iga,Q);CHKERRQ(ierr);
  PetscScalar mean;
  ierr = VecDot(Q,x,&mean);CHKERRQ(ierr);
  ierr = VecShift(x,-mean);CHKERRQ(ierr);
  ierr = VecDestroy(&Q);CHKERRQ(ierr);

  PetscScalar error;
  ierr = IGAComputeScalar(iga,x,1,&error,Error,NULL);CHKERRQ(ierr);
  error = PetscSqrtReal(PetscRealPart(error));

  if (print_error) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Error=%G\n",error);CHKERRQ(ierr);}
  if (check_error) {if (PetscRealPart(error)>1e-4) SETERRQ1(PETSC_COMM_WORLD,1,"Error=%G\n",error);}
  if (draw&&dim<3) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
