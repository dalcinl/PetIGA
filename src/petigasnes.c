#include "petiga.h"

extern PetscLogEvent IGA_FormFunction;
extern PetscLogEvent IGA_FormJacobian;

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeFunction"
PetscErrorCode IGAComputeFunction(IGA iga,Vec vecU,Vec vecF)
{
  IGAUserFunction   Function;
  void              *FunCtx;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,2);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,3);
  IGACheckSetUp(iga,1);
  IGACheckUserOp(iga,1,Function);
  Function = iga->userops->Function;
  FunCtx   = iga->userops->FunCtx;
  ierr = IGAFormFunction(iga,vecU,vecF,Function,FunCtx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormFunction"
PetscErrorCode IGAFormFunction(IGA iga,Vec vecU,Vec vecF,
                               IGAUserFunction Function,void *ctx)
{
  Vec               localU;
  const PetscScalar *arrayU;
  IGAElement        element;
  IGAPoint          point;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,2);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,3);
  IGACheckSetUp(iga,1);

  /* Clear global vector F*/
  ierr = VecZeroEntries(vecF);CHKERRQ(ierr);

  /* Get local vector U and array */
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Element loop */
  ierr = PetscLogEventBegin(IGA_FormFunction,iga,vecU,vecF,0);CHKERRQ(ierr);
  ierr = IGAGetElement(iga,&element);CHKERRQ(ierr);
  ierr = IGAElementBegin(element);CHKERRQ(ierr);
  while (IGAElementNext(element)) {
    PetscScalar *U, *F;
    ierr = IGAElementGetWorkVec(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&F);CHKERRQ(ierr);
    /* */
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementGetPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAPointBegin(point);CHKERRQ(ierr);
    while (IGAPointNext(point)) {
      PetscScalar *R;
      ierr = IGAPointGetWorkVec(point,&R);CHKERRQ(ierr);
      ierr = Function(point,U,R,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddVec(point,R,F);CHKERRQ(ierr);
    }
    /* */
    ierr = IGAElementFixFunction(element,F);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,F,vecF);CHKERRQ(ierr);
  }
  ierr = IGAElementEnd(element);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IGA_FormFunction,iga,vecU,vecF,0);CHKERRQ(ierr);

  /* Restore local vector U and array */
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Assemble global vector F */
  ierr = VecAssemblyBegin(vecF);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecF);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeJacobian"
PetscErrorCode IGAComputeJacobian(IGA iga,Vec vecU,Mat matJ)
{
  IGAUserJacobian   Jacobian;
  void              *JacCtx;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,2);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,3);
  IGACheckSetUp(iga,1);
  IGACheckUserOp(iga,1,Jacobian);
  Jacobian = iga->userops->Jacobian;
  JacCtx   = iga->userops->JacCtx;
  ierr = IGAFormJacobian(iga,vecU,matJ,Jacobian,JacCtx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormJacobian"
PetscErrorCode IGAFormJacobian(IGA iga,Vec vecU,Mat matJ,
                               IGAUserJacobian Jacobian,void *ctx)
{
  Vec               localU;
  const PetscScalar *arrayU;
  IGAElement        element;
  IGAPoint          point;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,2);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,3);
  IGACheckSetUp(iga,1);

  /* Clear global matrix J */
  ierr = MatZeroEntries(matJ);CHKERRQ(ierr);

  /* Get local vector U and array */
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Element Loop */
  ierr = PetscLogEventBegin(IGA_FormJacobian,iga,vecU,matJ,0);CHKERRQ(ierr);
  ierr = IGAGetElement(iga,&element);CHKERRQ(ierr);
  ierr = IGAElementBegin(element);CHKERRQ(ierr);
  while (IGAElementNext(element)) {
    PetscScalar *U, *J;
    ierr = IGAElementGetWorkVec(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkMat(element,&J);CHKERRQ(ierr);
    /* */
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementGetPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAPointBegin(point);CHKERRQ(ierr);
    while (IGAPointNext(point)) {
      PetscScalar *K;
      ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
      ierr = Jacobian(point,U,K,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddMat(point,K,J);CHKERRQ(ierr);
    }
    /* */
    ierr = IGAElementFixJacobian(element,J);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,J,matJ);CHKERRQ(ierr);
  }
  ierr = IGAElementEnd(element);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IGA_FormJacobian,iga,vecU,matJ,0);CHKERRQ(ierr);

  /* Restore local vector U and array */
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Assemble global matrix J*/
  ierr = MatAssemblyBegin(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASNESFormFunction"
PetscErrorCode IGASNESFormFunction(SNES snes,Vec U,Vec F,void *ctx)
{
  IGA            iga = (IGA)ctx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,2);
  PetscValidHeaderSpecific(F,VEC_CLASSID,3);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,4);
  ierr = IGAComputeFunction(iga,U,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASNESFormJacobian"
PetscErrorCode IGASNESFormJacobian(SNES snes,Vec U,Mat *J, Mat *P,MatStructure *m,void *ctx)
{
  IGA            iga = (IGA)ctx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,2);
  PetscValidPointer(J,3);
  PetscValidHeaderSpecific(*J,MAT_CLASSID,3);
  PetscValidPointer(P,4);
  PetscValidHeaderSpecific(*P,MAT_CLASSID,4);
  PetscValidPointer(m,5);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,6);
  ierr = IGAComputeJacobian(iga,U,*P);CHKERRQ(ierr);
  if (*J != * P) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *m = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateSNES"
PetscErrorCode IGACreateSNES(IGA iga, SNES *snes)
{
  MPI_Comm       comm;
  Vec            F;
  Mat            J;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(snes,2);

  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = SNESCreate(comm,snes);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&F);CHKERRQ(ierr);
  ierr = SNESSetFunction(*snes,F,IGASNESFormFunction,iga);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);

  ierr = IGACreateMat(iga,&J);CHKERRQ(ierr);
  ierr = SNESSetJacobian(*snes,J,J,IGASNESFormJacobian,iga);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  ierr = PetscObjectCompose((PetscObject)*snes,"IGA",(PetscObject)iga);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
