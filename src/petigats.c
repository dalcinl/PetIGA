#include "petiga.h"

PETSC_STATIC_INLINE
PetscBool IGAElementNextFormIFunction(IGAElement element,IGAFormIFunction *fun,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *fun = form->ops->IFunction;
  *ctx = form->ops->IFunCtx;
  return PETSC_TRUE;
}

PETSC_STATIC_INLINE
PetscBool IGAElementNextFormIJacobian(IGAElement element,IGAFormIJacobian *jac,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *jac = form->ops->IJacobian;
  *ctx = form->ops->IJacCtx;
  return PETSC_TRUE;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeIFunction"
PetscErrorCode IGAComputeIFunction(IGA iga,PetscReal dt,
                                   PetscReal a,Vec vecV,
                                   PetscReal t,Vec vecU,
                                   Vec vecF)
{
  Vec               localV;
  Vec               localU;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  IGAElement        element;
  IGAPoint          point;
  IGAFormIFunction  IFunction;
  void              *ctx;
  PetscScalar       *V,*U,*F,*R;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,7);
  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,IFunction);

  /* Clear global vector F */
  ierr = VecZeroEntries(vecF);CHKERRQ(ierr);

  /* Get local vectors V,U and arrays */
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormIFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkVec(element,&F);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,&V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,&U);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    /* FormIFunction loop */
    while (IGAElementNextFormIFunction(element,&IFunction,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkVec(point,&R);CHKERRQ(ierr);
        ierr = IFunction(point,dt,a,V,t,U,R,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddVec(point,R,F);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementFixFunction(element,F);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,F,vecF);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormIFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);

  /* Restore local vectors V,U and arrays */
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Assemble global vector F */
  ierr = VecAssemblyBegin(vecF);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecF);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeIJacobian"
PetscErrorCode IGAComputeIJacobian(IGA iga,PetscReal dt,
                                   PetscReal a,Vec vecV,
                                   PetscReal t,Vec vecU,
                                   Mat matJ)
{
  Vec               localV;
  Vec               localU;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  IGAElement        element;
  IGAPoint          point;
  IGAFormIJacobian  IJacobian;
  void              *ctx;
  PetscScalar       *V,*U,*J,*K;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,7);
  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,IJacobian);

  /* Clear global matrix J*/
  ierr = MatZeroEntries(matJ);CHKERRQ(ierr);

  /* Get local vectors V,U and arrays */
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormIJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);

  /* Element Loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkMat(element,&J);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,&V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,&U);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    /* FormIJacobian loop */
    while (IGAElementNextFormIJacobian(element,&IJacobian,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
        ierr = IJacobian(point,dt,a,V,t,U,K,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddMat(point,K,J);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementFixJacobian(element,J);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,J,matJ);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormIJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);

  /* Get local vectors V,U and arrays */
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Assemble global matrix J*/
  ierr = MatAssemblyBegin(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PETSC_STATIC_INLINE
PetscBool IGAElementNextFormIEFunction(IGAElement element,IGAFormIEFunction *fun,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *fun = form->ops->IEFunction;
  *ctx = form->ops->IEFunCtx;
  return PETSC_TRUE;
}

PETSC_STATIC_INLINE
PetscBool IGAElementNextFormIEJacobian(IGAElement element,IGAFormIEJacobian *jac,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *jac = form->ops->IEJacobian;
  *ctx = form->ops->IEJacCtx;
  return PETSC_TRUE;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeIEFunction"
PetscErrorCode IGAComputeIEFunction(IGA iga,PetscReal dt,
                                    PetscReal a, Vec vecV,
                                    PetscReal t, Vec vecU,
                                    PetscReal t0,Vec vecU0,
                                    Vec vecF)
{
  Vec               localV;
  Vec               localU;
  Vec               localU0;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  const PetscScalar *arrayU0;
  IGAElement        element;
  IGAPoint          point;
  IGAFormIEFunction IEFunction;
  void              *ctx;
  PetscScalar       *V,*U,*U0,*F,*R;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecU0,VEC_CLASSID,8);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,9);
  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,IEFunction);

  /* Clear global vector F */
  ierr = VecZeroEntries(vecF);CHKERRQ(ierr);

  /* Get local vectors V,U,U0 and arrays */
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU0,&localU0,&arrayU0);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormIFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkVec(element,&F);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,&V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,&U);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU0,&U0);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);  /* XXX */
    ierr = IGAElementFixValues(element,U0);CHKERRQ(ierr); /* XXX */
    /* FormIEFunction loop */
    while (IGAElementNextFormIEFunction(element,&IEFunction,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkVec(point,&R);CHKERRQ(ierr);
        ierr = IEFunction(point,dt,a,V,t,U,t0,U0,R,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddVec(point,R,F);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementFixFunction(element,F);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,F,vecF);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormIFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);

  /* Restore local vectors V,U,U0 and arrays */
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU0,&localU0,&arrayU0);CHKERRQ(ierr);

  /* Assemble global vector F */
  ierr = VecAssemblyBegin(vecF);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecF);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeIEJacobian"
PetscErrorCode IGAComputeIEJacobian(IGA iga,PetscReal dt,
                                    PetscReal a, Vec vecV,
                                    PetscReal t, Vec vecU,
                                    PetscReal t0,Vec vecU0,
                                    Mat matJ)
{
  Vec               localV;
  Vec               localU;
  Vec               localU0;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  const PetscScalar *arrayU0;
  IGAElement        element;
  IGAPoint          point;
  IGAFormIEJacobian IEJacobian;
  void              *ctx;
  PetscScalar       *V,*U,*U0,*J,*K;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecU0,VEC_CLASSID,8);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,9);
  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,IEJacobian);

  /* Clear global matrix J*/
  ierr = MatZeroEntries(matJ);CHKERRQ(ierr);

  /* Get local vectors V,U,U0 and arrays */
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU0,&localU0,&arrayU0);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormIJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);

  /* Element Loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkMat(element,&J);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,&V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,&U);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU0,&U0);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);  /* XXX */
    ierr = IGAElementFixValues(element,U0);CHKERRQ(ierr); /* XXX */
    /* FormIEJacobian loop */
    while (IGAElementNextFormIEJacobian(element,&IEJacobian,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
        ierr = IEJacobian(point,dt,a,V,t,U,t0,U0,K,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddMat(point,K,J);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementFixJacobian(element,J);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,J,matJ);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormIJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);

  /* Restore local vectors V,U,U0 and arrays */
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU0,&localU0,&arrayU0);CHKERRQ(ierr);

  /* Assemble global matrix J*/
  ierr = MatAssemblyBegin(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode IGATSFormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode IGATSFormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);

#undef  __FUNCT__
#define __FUNCT__ "IGATSFormIFunction"
PetscErrorCode IGATSFormIFunction(TS ts,PetscReal t,Vec U,Vec V,Vec F,void *ctx)
{
  IGA            iga = (IGA)ctx;
  PetscReal      dt,a=0;
  PetscReal      t0;
  Vec            U0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(F,VEC_CLASSID,5);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,6);
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (iga->form->ops->IEFunction) {
    ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
    ierr = TSGetSolution(ts,&U0);CHKERRQ(ierr);
    ierr = IGAComputeIEFunction(iga,dt,a,V,t,U,t0,U0,F);CHKERRQ(ierr);
  } else {
    ierr = IGAComputeIFunction(iga,dt,a,V,t,U,F);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGATSFormIJacobian"
PetscErrorCode IGATSFormIJacobian(TS ts,PetscReal t,Vec U,Vec V,PetscReal shift,Mat J,Mat P,void *ctx)
{
  IGA            iga = (IGA)ctx;
  PetscReal      dt,a=shift;
  PetscReal      t0;
  Vec            U0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(J,MAT_CLASSID,6);
  PetscValidHeaderSpecific(P,MAT_CLASSID,7);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,9);
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (iga->form->ops->IEJacobian) {
    ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
    ierr = TSGetSolution(ts,&U0);CHKERRQ(ierr);
    ierr = IGAComputeIEJacobian(iga,dt,a,V,t,U,t0,U0,P);CHKERRQ(ierr);
  } else {
    ierr = IGAComputeIJacobian(iga,dt,a,V,t,U,P);CHKERRQ(ierr);
  }
  if (J != P) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if PETSC_VERSION_LT(3,5,0)
PETSC_EXTERN PetscErrorCode IGATSFormIJacobian_Legacy(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
PetscErrorCode IGATSFormIJacobian_Legacy(TS ts,PetscReal t,Vec U,Vec V,PetscReal shift,Mat *J,Mat *P,MatStructure *m,void *ctx)
{*m = SAME_NONZERO_PATTERN; return IGATSFormIJacobian(ts,t,U,V,shift,*J,*P,ctx);}
#define IGATSFormIJacobian IGATSFormIJacobian_Legacy
#endif

#undef  __FUNCT__
#define __FUNCT__ "IGACreateTS"
/*@
   IGACreateTS - Creates a TS (time stepper) which uses the same
   communicators as the IGA.

   Logically collective on IGA

   Input Parameter:
.  iga - the IGA context

   Output Parameter:
.  ts - the TS

   Level: normal

.keywords: IGA, create, TS
@*/
PetscErrorCode IGACreateTS(IGA iga,TS *ts)
{
  MPI_Comm       comm;
  Vec            U;
  Vec            F;
  Mat            J;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(ts,2);

  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = TSCreate(comm,ts);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*ts,"IGA",(PetscObject)iga);CHKERRQ(ierr);
  ierr = IGASetOptionsHandlerTS(*ts);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = TSSetSolution(*ts,U);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&F);CHKERRQ(ierr);
  ierr = TSSetIFunction(*ts,F,IGATSFormIFunction,iga);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);

  ierr = IGACreateMat(iga,&J);CHKERRQ(ierr);
  ierr = TSSetIJacobian(*ts,J,J,IGATSFormIJacobian,iga);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
#undef  __FUNCT__
#define __FUNCT__ "IGA_OptionsHandler_TS"
static PetscErrorCode IGA_OptionsHandler_TS(PetscObject obj,void *ctx)
{
  TS             ts = (TS)obj;
  IGA            iga;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (PetscOptionsPublishCount != 1) PetscFunctionReturn(0);
  ierr = PetscObjectQuery((PetscObject)ts,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  if (!iga) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);

  PetscFunctionReturn(0);
}
static PetscErrorCode OptHdlDel(PetscObject obj,void *ctx){return 0;}
*/

#undef  __FUNCT__
#define __FUNCT__ "IGASetOptionsHandlerTS"
PetscErrorCode IGASetOptionsHandlerTS(TS ts)
{
  SNES           snes;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  /*ierr = PetscObjectAddOptionsHandler((PetscObject)ts,IGA_OptionsHandler_TS,OptHdlDel,NULL);CHKERRQ(ierr);*/
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = IGASetOptionsHandlerSNES(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
