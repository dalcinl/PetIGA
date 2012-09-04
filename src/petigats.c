#include "petiga.h"

extern PetscLogEvent IGA_FormFunction;
extern PetscLogEvent IGA_FormJacobian;

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeIFunction"
PetscErrorCode IGAComputeIFunction(IGA iga,PetscReal dt,
                                   PetscReal a,Vec vecV,
                                   PetscReal t,Vec vecU,
                                   Vec vecF)
{
  IGAUserIFunction  IFunction;
  void              *IFunCtx;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,5);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,7);
  IGACheckSetUp(iga,1);
  IGACheckUserOp(iga,1,IFunction);
  IFunction = iga->userops->IFunction;
  IFunCtx   = iga->userops->IFunCtx;
  ierr = IGAFormIFunction(iga,dt,a,vecV,t,vecU,vecF,IFunction,IFunCtx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormIFunction"
PetscErrorCode IGAFormIFunction(IGA iga,PetscReal dt,
                                PetscReal a,Vec vecV,
                                PetscReal t,Vec vecU,
                                Vec vecF,
                                IGAUserIFunction IFunction, void *ctx)
{
  Vec               localV;
  Vec               localU;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  IGAElement        element;
  IGAPoint          point;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,7);
  IGACheckSetUp(iga,1);

  /* Clear global vector F */
  ierr = VecZeroEntries(vecF);CHKERRQ(ierr);

  /* Get local vectors V,U and arrays */
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Element loop */
  ierr = PetscLogEventBegin(IGA_FormFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    PetscScalar *V, *U, *F;
    ierr = IGAElementGetWorkVal(element,&V);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVal(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&F);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
      PetscScalar *R;
      ierr = IGAPointGetWorkVec(point,&R);CHKERRQ(ierr);
      ierr = IFunction(point,dt,a,V,t,U,R,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddVec(point,R,F);CHKERRQ(ierr);
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    /* */
    ierr = IGAElementFixFunction(element,F);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,F,vecF);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IGA_FormFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);

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
  IGAUserIJacobian  IJacobian;
  void              *IJacCtx;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,5);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,7);
  IGACheckSetUp(iga,1);
  IGACheckUserOp(iga,1,IJacobian);
  IJacobian = iga->userops->IJacobian;
  IJacCtx   = iga->userops->IJacCtx;
  ierr = IGAFormIJacobian(iga,dt,a,vecV,t,vecU,matJ,IJacobian,IJacCtx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormIJacobian"
PetscErrorCode IGAFormIJacobian(IGA iga,PetscReal dt,
                                PetscReal a,Vec vecV,
                                PetscReal t,Vec vecU,
                                Mat matJ,
                                IGAUserIJacobian IJacobian,void *ctx)
{
  Vec               localV;
  Vec               localU;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  IGAElement        element;
  IGAPoint          point;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,5);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,7);
  IGACheckSetUp(iga,1);

  /* Clear global matrix J*/
  ierr = MatZeroEntries(matJ);CHKERRQ(ierr);

  /* Get local vectors V,U and arrays */
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Element Loop */
  ierr = PetscLogEventBegin(IGA_FormJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    PetscScalar *V, *U, *J;
    ierr = IGAElementGetWorkVal(element,&V);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVal(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkMat(element,&J);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
      PetscScalar *K;
      ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
      ierr = IJacobian(point,dt,a,V,t,U,K,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddMat(point,K,J);CHKERRQ(ierr);
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    /* */
    ierr = IGAElementFixJacobian(element,J);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,J,matJ);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IGA_FormJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);

  /* Get local vectors V,U and arrays */
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Assemble global matrix J*/
  ierr = MatAssemblyBegin(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeIEFunction"
PetscErrorCode IGAComputeIEFunction(IGA iga,PetscReal dt,
                                    PetscReal a,Vec vecV,
                                    PetscReal t,Vec vecU,
                                    PetscReal t0,Vec vecU0,
                                    Vec vecF)
{
  IGAUserIEFunction IEFunction;
  void              *IEFunCtx;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecU0,VEC_CLASSID,8);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,9);
  IGACheckSetUp(iga,1);
  IGACheckUserOp(iga,1,IEFunction);
  IEFunction = iga->userops->IEFunction;
  IEFunCtx   = iga->userops->IEFunCtx;
  ierr = IGAFormIEFunction(iga,dt,a,vecV,t,vecU,t0,vecU0,vecF,IEFunction,IEFunCtx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormIEFunction"
PetscErrorCode IGAFormIEFunction(IGA iga,PetscReal dt,
                                 PetscReal a,Vec vecV,
                                 PetscReal t,Vec vecU,
                                 PetscReal t0,Vec vecU0,
                                 Vec vecF,
                                 IGAUserIEFunction IEFunction,void *ctx)
{
  Vec               localV;
  Vec               localU;
  Vec               localU0;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  const PetscScalar *arrayU0;
  IGAElement        element;
  IGAPoint          point;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecU0,VEC_CLASSID,8);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,9);
  IGACheckSetUp(iga,1);

  /* Clear global vector F */
  ierr = VecZeroEntries(vecF);CHKERRQ(ierr);

  /* Get local vectors V,U,U0 and arrays */
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU0,&localU0,&arrayU0);CHKERRQ(ierr);

  /* Element loop */
  ierr = PetscLogEventBegin(IGA_FormFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    PetscScalar *V, *U, *U0, *F;
    ierr = IGAElementGetWorkVal(element,&V);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVal(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVal(element,&U0);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU0,U0);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U0);CHKERRQ(ierr); /* XXX */
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);  /* XXX */
    ierr = IGAElementGetWorkVec(element,&F);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
      PetscScalar *R;
      ierr = IGAPointGetWorkVec(point,&R);CHKERRQ(ierr);
      ierr = IEFunction(point,dt,a,V,t,U,t0,U0,R,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddVec(point,R,F);CHKERRQ(ierr);
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    /* */
    ierr = IGAElementFixFunction(element,F);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,F,vecF);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IGA_FormFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);

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
                                    PetscReal a,Vec vecV,
                                    PetscReal t,Vec vecU,
                                    PetscReal t0,Vec vecU0,
                                    Mat matJ)
{
  IGAUserIEJacobian  IEJacobian;
  void              *IEJacCtx;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,5);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,7);
  IGACheckSetUp(iga,1);
  IGACheckUserOp(iga,1,IEJacobian);
  IEJacobian = iga->userops->IEJacobian;
  IEJacCtx   = iga->userops->IEJacCtx;
  ierr = IGAFormIEJacobian(iga,dt,a,vecV,t,vecU,t0,vecU0,matJ,IEJacobian,IEJacCtx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormIEJacobian"
PetscErrorCode IGAFormIEJacobian(IGA iga,PetscReal dt,
                                 PetscReal a,Vec vecV,
                                 PetscReal t,Vec vecU,
                                 PetscReal t0,Vec vecU0,
                                 Mat matJ,
                                 IGAUserIEJacobian IEJacobian,void *ctx)
{
  Vec               localV;
  Vec               localU;
  Vec               localU0;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  const PetscScalar *arrayU0;
  IGAElement        element;
  IGAPoint          point;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,5);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecU0,VEC_CLASSID,7);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,0);
  IGACheckSetUp(iga,1);

  /* Clear global matrix J*/
  ierr = MatZeroEntries(matJ);CHKERRQ(ierr);

  /* Get local vectors V,U,U0 and arrays */
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU0,&localU0,&arrayU0);CHKERRQ(ierr);

  /* Element Loop */
  ierr = PetscLogEventBegin(IGA_FormJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    PetscScalar *V, *U, *U0, *J;
    ierr = IGAElementGetWorkVal(element,&V);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVal(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVal(element,&U0);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU0,U0);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U0);CHKERRQ(ierr); /* XXX */
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);  /* XXX */
    ierr = IGAElementGetWorkMat(element,&J);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
      PetscScalar *K;
      ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
      ierr = IEJacobian(point,dt,a,V,t,U,t0,U0,K,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddMat(point,K,J);CHKERRQ(ierr);
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    /* */
    ierr = IGAElementFixJacobian(element,J);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,J,matJ);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IGA_FormJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);

  /* Restore local vectors V,U,U0 and arrays */
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU0,&localU0,&arrayU0);CHKERRQ(ierr);

  /* Assemble global matrix J*/
  ierr = MatAssemblyBegin(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGATSFormIFunction"
PetscErrorCode IGATSFormIFunction(TS ts,PetscReal t,Vec U,Vec V,Vec F,void *ctx)
{
  IGA            iga = (IGA)ctx;
  PetscReal      dt,a=0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(F,VEC_CLASSID,5);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,6);
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (iga->userops->IEFunction) {
    PetscReal t0;
    Vec       U0;
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
PetscErrorCode IGATSFormIJacobian(TS ts,PetscReal t,Vec U,Vec V,PetscReal shift,Mat *J,Mat *P,MatStructure *m,void *ctx)
{
  IGA            iga = (IGA)ctx;
  PetscReal      dt,a=shift;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidPointer(J,6);
  PetscValidHeaderSpecific(*J,MAT_CLASSID,6);
  PetscValidPointer(P,7);
  PetscValidHeaderSpecific(*P,MAT_CLASSID,7);
  PetscValidPointer(m,8);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,9);
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (iga->userops->IEJacobian) {
    PetscReal t0;
    Vec       U0;
    ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
    ierr = TSGetSolution(ts,&U0);CHKERRQ(ierr);
    ierr = IGAComputeIEJacobian(iga,dt,a,V,t,U,t0,U0,*P);CHKERRQ(ierr);
  } else {
    ierr = IGAComputeIJacobian(iga,dt,a,V,t,U,*P);CHKERRQ(ierr);
  }
  if (*J != * P) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *m = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

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
PetscErrorCode IGACreateTS(IGA iga, TS *ts)
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

  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = TSSetSolution(*ts,U);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&F);CHKERRQ(ierr);
  ierr = TSSetIFunction(*ts,F,IGATSFormIFunction,iga);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);

  ierr = IGACreateMat(iga,&J);CHKERRQ(ierr);
  ierr = TSSetIJacobian(*ts,J,J,IGATSFormIJacobian,iga);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  ierr = PetscObjectCompose((PetscObject)*ts,"IGA",(PetscObject)iga);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
