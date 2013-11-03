#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "Scalar"
PetscErrorCode Scalar(IGAPoint p,const PetscScalar U[],PetscInt n,PetscScalar *S,void *ctx)
{
  PetscInt i;
  for (i=0; i<n; i++) S[i] = (PetscScalar)1.0;
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "Vector"
PetscErrorCode Vector(IGAPoint p,PetscScalar *F,void *ctx)
{
  PetscInt dof = p->dof;
  PetscInt nen = p->nen;
  PetscReal *N = p->shape[0];
  PetscInt a,i;
  for (a=0; a<nen; a++) {
    PetscReal Na = N[a];
    for (i=0; i<dof; i++)
      F[a*dof+i] = Na * 1;
  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "Matrix"
PetscErrorCode Matrix(IGAPoint p,PetscScalar *K,void *ctx)
{
  PetscInt dof = p->dof;
  PetscInt nen = p->nen;
  PetscReal *N = p->shape[0];
  PetscInt a,b,i,j;
  for (a=0; a<nen; a++) {
    PetscReal Na = N[a];
    for (b=0; b<nen; b++) {
      PetscReal Nb = N[b];
      for (i=0; i<dof; i++)
        for (j=0; j<dof; j++)
          if (i==j)
            K[a*dof*nen*dof+i*nen*dof+b*dof+j] = Na*Nb;
    }
  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "System"
PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt dof = p->dof;
  PetscInt nen = p->nen;
  PetscReal *N = p->shape[0];
  PetscInt a,b,i,j;
  for (a=0; a<nen; a++) {
    PetscReal Na = N[a];
    for (b=0; b<nen; b++) {
      PetscReal Nb = N[b];
      for (i=0; i<dof; i++)
        for (j=0; j<dof; j++)
          if (i==j)
            K[a*dof*nen*dof+i*nen*dof+b*dof+j] = Na*Nb;
    }
    for (i=0; i<dof; i++)
      F[a*dof+i] = Na * 1;
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscInt       dim,dof;
  IGA            iga;
  PetscScalar    s;
  PetscReal      xmin,xmax;
  Vec            b,x;
  Mat            A;
  KSP            ksp;
  SNES           snes;
  TS             ts;
  DM             dm;
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);

  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  if (dim < 1) {ierr = IGASetDim(iga,dim=3);CHKERRQ(ierr);}
  ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);
  if (dof < 1) {ierr = IGASetDof(iga,dof=1);CHKERRQ(ierr);}
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGAFormScalar(iga,x,1,&s,Scalar,0);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  ierr = IGASetUserSystem(iga,System,0);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);;CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1e-6,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = VecMin(x,0,&xmin);CHKERRQ(ierr);
  ierr = VecMax(x,0,&xmax);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  ierr = IGASetUserVector(iga,Vector,0);CHKERRQ(ierr);
  ierr = IGAComputeVector(iga,b);CHKERRQ(ierr);
  ierr = IGASetUserMatrix(iga,Matrix,0);CHKERRQ(ierr);
  ierr = IGAComputeMatrix(iga,A);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);;CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1e-6,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = VecMin(x,0,&xmin);CHKERRQ(ierr);
  ierr = VecMax(x,0,&xmax);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

  ierr = IGACreateSNES(iga,&snes);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);

  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = IGACreateElemDM(iga,dof,&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = IGACreateGeomDM(iga,dim,&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = IGACreateNodeDM(iga,1,&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);

  ierr = IGACreateWrapperDM(iga,&dm);CHKERRQ(ierr);
  ierr = DMIGAGetIGA(dm,&iga);CHKERRQ(ierr);
  ierr = DMIGASetIGA(dm,iga);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&x);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&b);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
#if PETSC_VERSION_LT(3,5,0)
  ierr = DMCreateMatrix(dm,NULL,&A);CHKERRQ(ierr);
#else
  ierr = DMCreateMatrix(dm,&A);CHKERRQ(ierr);
#endif
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);

  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  if ((xmax-xmin) > 1e-2) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Unexpected result: x_min=%G x_max=%G\n",
                       (PetscScalar)xmin,(PetscScalar)xmax);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
