#include "petiga.h"
#include "petscts2.h"

extern PetscErrorCode IGATSFormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
extern PetscErrorCode IGATSFormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);

extern PetscLogEvent IGA_FormFunction;
extern PetscLogEvent IGA_FormJacobian;

#undef  __FUNCT__
#define __FUNCT__ "IGAFormIFunction2"
PetscErrorCode IGAFormIFunction2(IGA iga,PetscReal dt,
                                 PetscReal a,Vec vecA,
                                 PetscReal v,Vec vecV,
                                 PetscReal t,Vec vecU,
                                 Vec vecF,
                                 IGAUserIFunction2 IFunction, void *ctx)
{
  Vec               localA;
  Vec               localV;
  Vec               localU;
  const PetscScalar *arrayA;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  IGAElement        element;
  IGAPoint          point;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecA,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,8);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,9);
  IGACheckSetUp(iga,1);

  /* Clear global vector F */
  ierr = VecZeroEntries(vecF);CHKERRQ(ierr);

  /* Get local vectors A,V,U and arrays */
  ierr = IGAGetLocalVecArray(iga,vecA,&localA,&arrayA);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Element loop */
  ierr = PetscLogEventBegin(IGA_FormFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    PetscScalar *A, *V, *U, *F;
    ierr = IGAElementGetWorkVal(element,&A);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVal(element,&V);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVal(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayA,A);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&F);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
      PetscScalar *R;
      ierr = IGAPointGetWorkVec(point,&R);CHKERRQ(ierr);
      ierr = IFunction(point,dt,a,A,v,V,t,U,R,ctx);CHKERRQ(ierr);
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
  ierr = IGARestoreLocalVecArray(iga,vecA,&localA,&arrayA);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Assemble global vector F */
  ierr = VecAssemblyBegin(vecF);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecF);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormIJacobian2"
PetscErrorCode IGAFormIJacobian2(IGA iga,PetscReal dt,
                                PetscReal a,Vec vecA,
                                PetscReal v,Vec vecV,
                                PetscReal t,Vec vecU,
                                Mat matJ,
                                IGAUserIJacobian2 IJacobian,void *ctx)
{
  Vec               localA;
  Vec               localV;
  Vec               localU;
  const PetscScalar *arrayA;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  IGAElement        element;
  IGAPoint          point;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecA,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,8);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,9);
  IGACheckSetUp(iga,1);

  /* Clear global matrix J*/
  ierr = MatZeroEntries(matJ);CHKERRQ(ierr);

  /* Get local vectors A,V,U and arrays */
  ierr = IGAGetLocalVecArray(iga,vecA,&localA,&arrayA);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Element Loop */
  ierr = PetscLogEventBegin(IGA_FormJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    PetscScalar *A, *V, *U, *J;
    ierr = IGAElementGetWorkVal(element,&A);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVal(element,&V);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVal(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayA,A);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkMat(element,&J);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
      PetscScalar *K;
      ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
      ierr = IJacobian(point,dt,a,A,v,V,t,U,K,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddMat(point,K,J);CHKERRQ(ierr);
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    /* */
    ierr = IGAElementFixJacobian(element,J);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,J,matJ);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IGA_FormJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);

  /* Restore local vectors A,V,U and arrays */
  ierr = IGARestoreLocalVecArray(iga,vecA,&localA,&arrayA);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Assemble global matrix J*/
  ierr = MatAssemblyBegin(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeIFunction2"
PetscErrorCode IGAComputeIFunction2(IGA iga,PetscReal dt,
                                    PetscReal a,Vec vecA,
                                    PetscReal v,Vec vecV,
                                    PetscReal t,Vec vecU,
                                    Vec vecF)
{
  IGAUserIFunction2 IFunction;
  void              *IFunCtx;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecA,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,8);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,9);
  IGACheckSetUp(iga,1);
  IGACheckUserOp(iga,1,IFunction2);
  IFunction = iga->userops->IFunction2;
  IFunCtx   = iga->userops->IFunCtx;
  ierr = IGAFormIFunction2(iga,dt,a,vecA,v,vecV,t,vecU,vecF,IFunction,IFunCtx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeIJacobian2"
PetscErrorCode IGAComputeIJacobian2(IGA iga,PetscReal dt,
                                    PetscReal a,Vec vecA,
                                    PetscReal v,Vec vecV,
                                    PetscReal t,Vec vecU,
                                    Mat matJ)
{
  IGAUserIJacobian2 IJacobian;
  void              *IJacCtx;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecA,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,8);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,9);
  IGACheckSetUp(iga,1);
  IGACheckUserOp(iga,1,IJacobian2);
  IJacobian = iga->userops->IJacobian2;
  IJacCtx   = iga->userops->IJacCtx;
  ierr = IGAFormIJacobian2(iga,dt,a,vecA,v,vecV,t,vecU,matJ,IJacobian,IJacCtx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGATSFormIFunction2"
PetscErrorCode IGATSFormIFunction2(TS ts,PetscReal t,Vec U,Vec V,Vec A,Vec F,void *ctx)
{
  IGA            iga = (IGA)ctx;
  PetscReal      dt,a=0,v=0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(A,VEC_CLASSID,5);
  PetscValidHeaderSpecific(F,VEC_CLASSID,6);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,7);
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = IGAComputeIFunction2(iga,dt,a,A,v,V,t,U,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGATSFormIJacobian2"
PetscErrorCode IGATSFormIJacobian2(TS ts,PetscReal t,Vec U,Vec V,Vec A,PetscReal shiftV,PetscReal shiftA,Mat *J,Mat *P,MatStructure *m,void *ctx)
{
  IGA            iga = (IGA)ctx;
  PetscReal      dt,a=shiftA,v=shiftV;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(A,VEC_CLASSID,5);
  PetscValidPointer(J,8);
  PetscValidHeaderSpecific(*J,MAT_CLASSID,8);
  PetscValidPointer(P,9);
  PetscValidHeaderSpecific(*P,MAT_CLASSID,9);
  PetscValidPointer(m,10);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,10);
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = IGAComputeIJacobian2(iga,dt,a,A,v,V,t,U,*P);CHKERRQ(ierr);
  if (*J != *P) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *m = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateTS2"
/*@
   IGACreateTS2 - Creates a TS (time stepper) which uses the same
   communicators as the IGA.

   Logically collective on IGA

   Input Parameter:
.  iga - the IGA context

   Output Parameter:
.  ts - the TS

   Level: normal

.keywords: IGA, create, TS
@*/
PetscErrorCode IGACreateTS2(IGA iga, TS *ts)
{
  MPI_Comm       comm;
  Vec            U,V;
  Vec            F;
  Mat            J;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(ts,2);

  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = TSCreate(comm,ts);CHKERRQ(ierr);
  ierr = TSSetType(*ts,TSALPHA2);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&V);CHKERRQ(ierr);
  ierr = TSSetSolution2(*ts,U,V);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&V);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&F);CHKERRQ(ierr);
  ierr = TSSetIFunction (*ts,F,IGATSFormIFunction ,iga);CHKERRQ(ierr);
  ierr = TSSetIFunction2(*ts,F,IGATSFormIFunction2,iga);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);

  ierr = IGACreateMat(iga,&J);CHKERRQ(ierr);
  ierr = TSSetIJacobian (*ts,J,J,IGATSFormIJacobian, iga);CHKERRQ(ierr);
  ierr = TSSetIJacobian2(*ts,J,J,IGATSFormIJacobian2,iga);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  ierr = PetscObjectCompose((PetscObject)*ts,"IGA",(PetscObject)iga);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
