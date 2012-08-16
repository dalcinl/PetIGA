#include "petiga.h"

EXTERN_C_BEGIN
extern PetscErrorCode Bratu_Function(IGAPoint,const PetscScalar U[],PetscScalar F[],void *);
extern PetscErrorCode Bratu_Jacobian(IGAPoint,const PetscScalar U[],PetscScalar J[],void *);
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);

  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  IGABoundary bnd;
  PetscInt dir,side;
  for (dir=0; dir<2; dir++) {
    for (side=0; side<2; side++) {
      ierr = IGAGetBoundary(iga,dir,side,&bnd);CHKERRQ(ierr);
      ierr = IGABoundarySetValue(bnd,0,0.0);CHKERRQ(ierr);
    }
  }

  IGAUserFunction Function = Bratu_Function;
  IGAUserJacobian Jacobian = Bratu_Jacobian;
  ierr = IGASetUserFunction(iga,*Function,0);CHKERRQ(ierr);
  ierr = IGASetUserJacobian(iga,*Jacobian,0);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);

  PetscInt dim;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  if (dim < 1) {ierr = IGASetDim(iga,dim=2);CHKERRQ(ierr);}
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Vec x;
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  SNES snes;
  ierr = IGACreateSNES(iga,&snes);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSolve(snes,0,x);CHKERRQ(ierr);

  PetscBool draw = PETSC_FALSE;
  ierr = PetscOptionsGetBool(0,"-draw",&draw,0);CHKERRQ(ierr);
  if (draw && dim < 3) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  PetscBool flag = PETSC_FALSE;
  PetscReal secs = -1;
  ierr = PetscOptionsHasName(0,"-pause",&flag);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(0,"-pause",&secs,0);CHKERRQ(ierr);
  if (flag) {ierr = PetscSleep(secs);CHKERRQ(ierr);}

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
