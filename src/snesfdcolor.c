#include <petscsnes.h>
#if PETSC_VERSION_(3,2,0)
#include <private/petscimpl.h>
#else
#include <petsc-private/petscimpl.h>
#endif

#if PETSC_VERSION_(3,3,0) || PETSC_VERSION_(3,2,0)
#define SNESComputeJacobianDefaultColor SNESDefaultComputeJacobianColor
#endif

PETSC_EXTERN PetscErrorCode SNESSetUpFDColoring(SNES);

#undef __FUNCT__
#define __FUNCT__ "MatFDColoringSetOptionsPrefix"
static PetscErrorCode MatFDColoringSetOptionsPrefix(MatFDColoring fdc, const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fdc,MAT_FDCOLORING_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)fdc,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUpFDColoring"
PetscErrorCode SNESSetUpFDColoring(SNES snes)
{
  const char*    prefix = PETSC_NULL;
  Vec            f = PETSC_NULL;
  PetscErrorCode (*fun)(SNES,Vec,Vec,void*) = PETSC_NULL;
  void*          funP = PETSC_NULL;
  Mat            A = PETSC_NULL, B = PETSC_NULL;
  PetscErrorCode (*jac)(SNES,Vec,Mat*,Mat*,MatStructure*,void*) = PETSC_NULL;
  void*          jacP = PETSC_NULL;
  ISColoring     iscoloring = PETSC_NULL;
  MatFDColoring  fdcoloring = PETSC_NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);

  ierr = PetscObjectQuery((PetscObject)snes,"fdcoloring",(PetscObject*)&fdcoloring);CHKERRQ(ierr);
  if (fdcoloring) PetscFunctionReturn(0);

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

  ierr = MatGetColoring((B?B:A),MATCOLORINGSL,&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringCreate((B?B:A),iscoloring,&fdcoloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))fun,funP);
  ierr = MatFDColoringSetOptionsPrefix(fdcoloring,prefix);CHKERRQ(ierr);
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
  PetscBool      fdc = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscOptionsBool("-snes_fd_color","Use colored finite differences to compute Jacobian","SNESSetUpFDColoring",fdc,&fdc,PETSC_NULL);CHKERRQ(ierr);
  if (fdc) {ierr = SNESSetUpFDColoring(snes);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
EXTERN_C_END
