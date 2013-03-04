#include "petiga.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscInt       dim,dof;
  IGA            iga;
  Vec            v;
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

  ierr = IGACreateElemDM(iga,dof,&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = IGACreateGeomDM(iga,dim,&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = IGACreateNodeDM(iga,1,&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&v);CHKERRQ(ierr);
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = IGACreateSNES(iga,&snes);CHKERRQ(ierr);
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);

  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
