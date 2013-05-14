#include "petiga.h"

PETSC_STATIC_INLINE
PetscBool IGAElementNextUserFunction(IGAElement element,IGAUserFunction *fun,void **ctx)
{
  IGAUserOps ops;
  while (IGAElementNextUserOps(element,&ops) && !ops->Function);
  if (!ops) return PETSC_FALSE;
  *fun = ops->Function;
  *ctx = ops->FunCtx;
  return PETSC_TRUE;
}

PETSC_STATIC_INLINE
PetscBool IGAElementNextUserJacobian(IGAElement element,IGAUserJacobian *jac,void **ctx)
{
  IGAUserOps ops;
  while (IGAElementNextUserOps(element,&ops) && !ops->Jacobian);
  if (!ops) return PETSC_FALSE;
  *jac = ops->Jacobian;
  *ctx = ops->JacCtx;
  return PETSC_TRUE;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeFunction"
PetscErrorCode IGAComputeFunction(IGA iga,Vec vecU,Vec vecF)
{
  Vec               localU;
  const PetscScalar *arrayU;
  IGAElement        element;
  IGAPoint          point;
  IGAUserFunction   Function;
  void              *ctx;
  PetscScalar       *U,*F,*R;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,2);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,3);
  IGACheckSetUp(iga,1);
  IGACheckUserOp(iga,1,Function);

  /* Clear global vector F*/
  ierr = VecZeroEntries(vecF);CHKERRQ(ierr);

  /* Get local vector U and array */
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormFunction,iga,vecU,vecF,0);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkVal(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&F);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    /* UserFunction loop */
    while (IGAElementNextUserFunction(element,&Function,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkVec(point,&R);CHKERRQ(ierr);
        ierr = Function(point,U,R,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddVec(point,R,F);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementFixFunction(element,F);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,F,vecF);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

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
  Vec               localU;
  const PetscScalar *arrayU;
  IGAElement        element;
  IGAPoint          point;
  IGAUserJacobian   Jacobian;
  void              *ctx;
  PetscScalar       *U,*J,*K;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,2);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,3);
  IGACheckSetUp(iga,1);
  IGACheckUserOp(iga,1,Jacobian);

  /* Clear global matrix J */
  ierr = MatZeroEntries(matJ);CHKERRQ(ierr);

  /* Get local vector U and array */
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormJacobian,iga,vecU,matJ,0);CHKERRQ(ierr);

  /* Element Loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkVal(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkMat(element,&J);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    /* UserJacobian loop */
    while (IGAElementNextUserJacobian(element,&Jacobian,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
        ierr = Jacobian(point,U,K,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddMat(point,K,J);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementFixJacobian(element,J);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,J,matJ);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormJacobian,iga,vecU,matJ,0);CHKERRQ(ierr);

  /* Restore local vector U and array */
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Assemble global matrix J*/
  ierr = MatAssemblyBegin(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode IGASNESFormFunction(SNES,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode IGASNESFormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

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
  ierr = PetscObjectCompose((PetscObject)*snes,"IGA",(PetscObject)iga);CHKERRQ(ierr);
  ierr = IGASetOptionsHandlerSNES(*snes);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&F);CHKERRQ(ierr);
  ierr = SNESSetFunction(*snes,F,IGASNESFormFunction,iga);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);

  ierr = IGACreateMat(iga,&J);CHKERRQ(ierr);
  ierr = SNESSetJacobian(*snes,J,J,IGASNESFormJacobian,iga);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
#undef  __FUNCT__
#define __FUNCT__ "IGA_OptionsHandler_SNES"
static PetscErrorCode IGA_OptionsHandler_SNES(PetscObject obj,void *ctx)
{
  SNES            snes = (SNES)obj;
  IGA            iga;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (PetscOptionsPublishCount != 1) PetscFunctionReturn(0);
  ierr = PetscObjectQuery((PetscObject)snes,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  if (!iga) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);

  PetscFunctionReturn(0);
}
static PetscErrorCode OptHdlDel(PetscObject obj,void *ctx){return 0;}
*/

#undef  __FUNCT__
#define __FUNCT__ "IGASetOptionsHandlerSNES"
PetscErrorCode IGASetOptionsHandlerSNES(SNES snes)
{
  KSP            ksp;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  /*ierr = PetscObjectAddOptionsHandler((PetscObject)snes,IGA_OptionsHandler_SNES,OptHdlDel,PETSC_NULL);CHKERRQ(ierr);*/
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = IGASetOptionsHandlerKSP(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
