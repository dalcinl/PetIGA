#include <petscsnes.h>
#include <petsc-private/petscimpl.h>

#if PETSC_VERSION_LE(3,3,0)
#undef  __FUNCT__
#define __FUNCT__ "SNESComputeJacobianDefaultColor"
PETSC_EXTERN PetscErrorCode SNESComputeJacobianDefaultColor(SNES snes,Vec x,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  Vec            f = NULL;
  PetscErrorCode (*fun)(SNES,Vec,Vec,void*) = NULL;
  void*          funP = NULL;
  ISColoring     iscoloring = NULL;
  MatFDColoring  color = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = SNESGetFunction(snes,&f,&fun,&funP);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)*B,"SNESMatFDColoring",(PetscObject*)&color);CHKERRQ(ierr);
  if (!color) {
    ierr = MatGetColoring(*B,MATCOLORINGSL,&iscoloring);CHKERRQ(ierr);
    ierr = MatFDColoringCreate(*B,iscoloring,&color);CHKERRQ(ierr);
    ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
    ierr = MatFDColoringSetFunction(color,(PetscErrorCode(*)(void))fun,(void*)funP);CHKERRQ(ierr);
    ierr = MatFDColoringSetFromOptions(color);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)*B,"SNESMatFDColoring",(PetscObject)color);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)color);CHKERRQ(ierr);
  }
  ierr = SNESDefaultComputeJacobianColor(snes,x,J,B,flag,(void*)color);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_FDColor"
PetscErrorCode SNESSetFromOptions_FDColor(SNES snes)
{
  PetscBool      flg = PETSC_FALSE;
  void           *ctx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsBool("-snes_fd_color","Use finite differences with coloring to compute Jacobian","SNESComputeJacobianDefaultColor",flg,&flg,NULL);CHKERRQ(ierr);
  if (PetscOptionsPublishCount != 1) PetscFunctionReturn(0);
  if (flg) {
    ierr = SNESGetFunction(snes,NULL,NULL,&ctx);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,NULL,NULL,SNESComputeJacobianDefaultColor,NULL);CHKERRQ(ierr);
    ierr = PetscInfo(snes,"Setting default finite difference coloring Jacobian matrix\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
