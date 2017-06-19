#include "petiga.h"

PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  const PetscReal *N0,(*N1)[2];
  IGAPointGetShapeFuns(p,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(p,1,(const PetscReal**)&N1);
  PetscInt a,b,nen=p->nen;
  for (a=0; a<nen; a++) {
    PetscReal Na   = N0[a];
    PetscReal Na_x = N1[a][0];
    PetscReal Na_y = N1[a][1];
    for (b=0; b<nen; b++) {
      PetscReal Nb_x = N1[b][0];
      PetscReal Nb_y = N1[b][1];
      K[a*nen+b] = Na_x*Nb_x + Na_y*Nb_y;
    }
    F[a] = Na * 1.0;
  }
  return 0;
}

int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  
  PetscInt dir,side;
  PetscScalar value = 1.0;
  for (dir=0; dir<2; dir++) {
    for (side=0; side<2; side++) {
      ierr = IGASetBoundaryValue(iga,dir,side,0,value);CHKERRQ(ierr);
    }
  }

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetFormSystem(iga,System,NULL);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  
  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
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
