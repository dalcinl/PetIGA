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

#undef  __FUNCT__
#define __FUNCT__ "System"
PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen,dim;
  IGAPointGetSizes(p,0,&nen,0);
  IGAPointGetDims(p,&dim,0,0);

  PetscReal x[3] = {0,0,0};
  IGAPointFormPoint(p,x);
  PetscReal f = Forcing(x);

  const PetscReal *N0,*N1;
  IGAPointGetShapeFuns(p,0,&N0);
  IGAPointGetShapeFuns(p,1,&N1);

  PetscInt a,b,i;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      PetscScalar Kab = 0.0;
      for (i=0; i<dim; i++)
        Kab += N1[a*dim+i]*N1[b*dim+i];
      K[a*nen+b] = Kab;
    }
    F[a] = N0[a]*f;
  }
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

  PetscInt dim = 3;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Neumann Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","dimension",__FILE__,dim,&dim,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);

  IGABoundary bnd;
  PetscInt dir,side;
  for (dir=0; dir<dim; dir++) {
    for (side=0; side<2; side++) {
      PetscScalar load = -pi;
      ierr = IGAGetBoundary(iga,dir,side,&bnd);CHKERRQ(ierr);
      ierr = IGABoundarySetLoad(bnd,0,load);CHKERRQ(ierr);
    }
  }

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetUserSystem(iga,System,PETSC_NULL);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  {
    MPI_Comm comm;
    MatNullSpace nsp;
    ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(comm,PETSC_TRUE,0,PETSC_NULL,&nsp);CHKERRQ(ierr);
    ierr = KSPSetNullSpace(ksp,nsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);
  }
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  {
    PetscReal vmin;
    ierr = VecMin(x,0,&vmin);CHKERRQ(ierr);
    ierr = VecShift(x,-vmin);CHKERRQ(ierr);
  }

  PetscScalar se = 0;
  ierr = IGAFormScalar(iga,x,1,&se,Error,PETSC_NULL);CHKERRQ(ierr);
  PetscReal e = PetscSqrtReal(PetscRealPart(se));

  PetscBool error = PETSC_FALSE;
  ierr = PetscOptionsGetBool(0,"-error",&error,0);CHKERRQ(ierr);
  if (error) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Error=%G\n",e);CHKERRQ(ierr);}
  else if (e>1e-4) SETERRQ1(PETSC_COMM_WORLD,1,"Error=%G\n",e);

  if (dim < 3) {
    PetscBool draw = PETSC_FALSE;
    ierr = PetscOptionsGetBool(0,"-draw",&draw,0);CHKERRQ(ierr);
    if (draw) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
