#include <petscsnes.h>
#include <petsc-private/petscimpl.h>

#if PETSC_VERSION_LE(3,3,0)
#define SNESComputeJacobianDefaultColor SNESDefaultComputeJacobianColor
#endif

PETSC_EXTERN PetscErrorCode SNESSetUpFDColoring(SNES);

#undef __FUNCT__
#define __FUNCT__ "SNESSetUpFDColoring"
PetscErrorCode SNESSetUpFDColoring(SNES snes)
{
  const char*    prefix = NULL;
  Vec            f = NULL;
  PetscErrorCode (*fun)(SNES,Vec,Vec,void*) = NULL;
  void*          funP = NULL;
  Mat            A = NULL, B = NULL;
  PetscErrorCode (*jac)(SNES,Vec,Mat*,Mat*,MatStructure*,void*) = NULL;
  void*          jacP = NULL;
  ISColoring     iscoloring = NULL;
  MatFDColoring  fdcoloring = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);

  ierr = SNESGetOptionsPrefix(snes,&prefix);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes,&f,&fun,&funP);CHKERRQ(ierr);
  ierr = SNESGetJacobian(snes,&A,&B,&jac,&jacP);CHKERRQ(ierr);
  if (!fun) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"SNESSetFunction() must be called first");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }
  if (!A && !B) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"SNESSetJacobian() must be called first");
    PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
  }
  ierr = PetscObjectQuery((PetscObject)snes,"fdcoloring",(PetscObject*)&fdcoloring);CHKERRQ(ierr);
  if (fdcoloring && fdcoloring == (MatFDColoring)jacP) PetscFunctionReturn(0);

  ierr = MatGetColoring((B?B:A),MATCOLORINGSL,&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringCreate((B?B:A),iscoloring,&fdcoloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))fun,funP);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)fdcoloring,prefix);CHKERRQ(ierr);
  ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)snes,"fdcoloring",(PetscObject)fdcoloring);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,A,B,SNESComputeJacobianDefaultColor,fdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringDestroy(&fdcoloring);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_FDColoring"
PetscErrorCode SNESSetFromOptions_FDColoring(SNES snes)
{
  static PetscBool fdc = PETSC_FALSE;
  PetscBool        opt;
  PetscErrorCode   ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsBool("-snes_fd_color","Use colored finite differences to compute Jacobian","SNESSetUpFDColoring",fdc,&fdc,NULL);CHKERRQ(ierr);
  if (PetscOptionsPublishCount != 1) PetscFunctionReturn(0);
  opt = fdc; fdc = PETSC_FALSE;
  if (opt) {ierr = SNESSetUpFDColoring(snes);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
EXTERN_C_END
