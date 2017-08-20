#include "petiga.h"

#define pi M_PI

PetscReal Solution(PetscReal x[3])
{
  return sin(2*pi*x[0]) + sin(2*pi*x[1]) + sin(2*pi*x[2]);
}

PetscReal Forcing(PetscReal x[3])
{
  return 4*pi*pi * Solution(x);
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

PetscErrorCode SystemGalerkin(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;
  const PetscReal *N0        = (typeof(N0)) p->shape[0];
  const PetscReal (*N1)[dim] = (typeof(N1)) p->shape[1];

  PetscReal x[3] = {0,0,0};
  IGAPointFormGeomMap(p,x);
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

PetscErrorCode SystemCollocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;
  const PetscReal (*N2)[dim][dim] = (typeof(N2)) p->shape[2];

  PetscReal x[3] = {0,0,0};
  IGAPointFormGeomMap(p,x);
  PetscReal f = Forcing(x);

  PetscInt a;
  for (a=0; a<nen; a++)
    K[a] += -DEL2(dim,N2[a]);
  F[0] = f;
  return 0;
}

PetscErrorCode MassVector(IGAPoint p,PetscScalar *V,void *ctx)
{
  PetscInt a,nen = p->nen;
  const PetscReal *N0 = (typeof(N0)) p->shape[0];
  for (a=0; a<nen; a++) V[a] = N0[a];
  return 0;
}

PetscErrorCode Exact(IGAPoint p,PetscInt order,PetscScalar value[],void *ctx)
{
  PetscReal x[3] = {0,0,0};
  IGAPointFormGeomMap(p,x);
  value[0] = Solution(x);
  return 0;
}

int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscBool print_error = PETSC_FALSE;
  PetscBool check_error = PETSC_FALSE;
  PetscBool save = PETSC_FALSE;
  PetscBool draw = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Neumann Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-print_error","Prints the error of the solution",__FILE__,print_error,&print_error,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-check_error","Checks the error of the solution",__FILE__,check_error,&check_error,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-save","Save the solution to file",__FILE__,save,&save,NULL);CHKERRQ(ierr);
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
    ierr = MatSetNullSpace(A,nsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);
  }

  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  if (!iga->collocation) {
    Vec Q; PetscScalar mean;
    ierr = IGACreateVec(iga,&Q);CHKERRQ(ierr);
    ierr = IGASetFormVector(iga,MassVector,NULL);CHKERRQ(ierr);
    ierr = IGAComputeVector(iga,Q);CHKERRQ(ierr);
    ierr = VecDot(Q,x,&mean);CHKERRQ(ierr);
    ierr = VecShift(x,-mean);CHKERRQ(ierr);
    ierr = VecDestroy(&Q);CHKERRQ(ierr);
  } else {
    MatNullSpace nsp;
    ierr = MatGetNullSpace(A,&nsp);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nsp,x);CHKERRQ(ierr);
  }

  PetscReal error;
  ierr = IGAComputeErrorNorm(iga,0,x,Exact,&error,NULL);CHKERRQ(ierr);

  if (print_error) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Error=%g\n",(double)error);CHKERRQ(ierr);}
  if (check_error) {if (error>1e-3) SETERRQ1(PETSC_COMM_WORLD,1,"Error=%g\n",(double)error);}
  if (draw&&dim<3) {ierr = IGADrawVec(iga,x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  if (save) {ierr = IGAWrite   (iga,  "Neumann-geometry.dat");CHKERRQ(ierr);}
  if (save) {ierr = IGAWriteVec(iga,x,"Neumann-solution.dat");CHKERRQ(ierr);}

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
