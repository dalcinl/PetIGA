#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "System"
PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscReal *N0 = p->shape[0];
  PetscReal *N1 = p->shape[1];
  PetscInt a,b,nen=p->nen;
  for (a=0; a<nen; a++) {
    PetscReal Na   = N0[a];
    PetscReal Na_x = N1[a];
    for (b=0; b<nen; b++) {
      PetscReal Nb_x = N1[b];
      K[a*nen+b] = Na_x * Nb_x;
    }
    F[a] = Na * 1.0;
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscInt N=16, p=2, C=PETSC_DECIDE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Poisson1D Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N","number of elements (along one dimension)",__FILE__,N,&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p","polynomial order",__FILE__,p,&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C","global continuity order",__FILE__,C,&C,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (C == PETSC_DECIDE) C = p-1;

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,1);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);

  IGAAxis axis;
  ierr = IGAGetAxis(iga,0,&axis);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis,N,-1.0,+1.0,C);CHKERRQ(ierr);

  IGABoundary left;
  IGABoundary right;
  PetscScalar value = 1.0;
  ierr = IGAGetBoundary(iga,0,0,&left);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(left,0,value);CHKERRQ(ierr);
  ierr = IGAGetBoundary(iga,0,1,&right);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(right,0,value);CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetUserSystem(iga,System,PETSC_NULL);CHKERRQ(ierr);
  ierr = IGAFormSystem(iga,A,b);CHKERRQ(ierr);
  
  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
