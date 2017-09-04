#include "petiga.h"

PETSC_STATIC_INLINE
PetscBool IGAElementNextFormIFunction(IGAElement element,IGAFormI2Function *fun,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *fun = form->ops->I2Function;
  *ctx = form->ops->IFunCtx;
  return PETSC_TRUE;
}

PETSC_STATIC_INLINE
PetscBool IGAElementNextFormIJacobian(IGAElement element,IGAFormI2Jacobian *jac,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *jac = form->ops->I2Jacobian;
  *ctx = form->ops->IJacCtx;
  return PETSC_TRUE;
}

PetscErrorCode IGAComputeI2Function(IGA iga,
                                    PetscReal a,Vec vecA,
                                    PetscReal v,Vec vecV,
                                    PetscReal t,Vec vecU,
                                    Vec vecF)
{
  Vec               localA;
  Vec               localV;
  Vec               localU;
  const PetscScalar *arrayA;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  IGAElement        element;
  IGAPoint          point;
  IGAFormI2Function IFunction;
  void              *ctx;
  PetscScalar       *A,*V,*U,*F,*R;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecA,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,8);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,9);
  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,I2Function);

  /* Clear global vector F */
  ierr = VecZeroEntries(vecF);CHKERRQ(ierr);

  /* Get local vectors A,V,U and arrays */
  ierr = IGAGetLocalVecArray(iga,vecA,&localA,&arrayA);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormIFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkVec(element,&F);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayA,&A);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,&V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,&U);CHKERRQ(ierr);
    ierr = IGAElementDelValues(element,A);CHKERRQ(ierr);
    ierr = IGAElementDelValues(element,V);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    while (IGAElementNextFormIFunction(element,&IFunction,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkVec(point,&R);CHKERRQ(ierr);
        ierr = IFunction(point,a,A,v,V,t,U,R,ctx);CHKERRQ(ierr);
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
  ierr = IGARestoreLocalVecArray(iga,vecA,&localA,&arrayA);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Assemble global vector F */
  ierr = VecAssemblyBegin(vecF);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAComputeI2Jacobian(IGA iga,
                                    PetscReal a,Vec vecA,
                                    PetscReal v,Vec vecV,
                                    PetscReal t,Vec vecU,
                                    Mat matJ)
{
  Vec               localA;
  Vec               localV;
  Vec               localU;
  const PetscScalar *arrayA;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  IGAElement        element;
  IGAPoint          point;
  IGAFormI2Jacobian IJacobian;
  void              *ctx;
  PetscScalar       *A,*V,*U,*J,*K;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecA,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,8);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,9);
  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,I2Jacobian);
  IJacobian = iga->form->ops->I2Jacobian;
  ctx       = iga->form->ops->IJacCtx;

  /* Clear global matrix J*/
  ierr = MatZeroEntries(matJ);CHKERRQ(ierr);

  /* Get local vectors A,V,U and arrays */
  ierr = IGAGetLocalVecArray(iga,vecA,&localA,&arrayA);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormIJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);

  /* Element Loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkMat(element,&J);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayA,&A);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,&V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,&U);CHKERRQ(ierr);
    ierr = IGAElementDelValues(element,A);CHKERRQ(ierr);
    ierr = IGAElementDelValues(element,V);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    while (IGAElementNextFormIJacobian(element,&IJacobian,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
        ierr = IJacobian(point,a,A,v,V,t,U,K,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddMat(point,K,J);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementFixJacobian(element,J);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,J,matJ);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormIJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);

  /* Restore local vectors A,V,U and arrays */
  ierr = IGARestoreLocalVecArray(iga,vecA,&localA,&arrayA);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Assemble global matrix J*/
  ierr = MatAssemblyBegin(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGATSFormI2Function(TS ts,PetscReal t,Vec U,Vec V,Vec A,Vec F,void *ctx)
{
  IGA            iga = (IGA)ctx;
  PetscReal      a=0,v=0;
  PetscReal      dt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(A,VEC_CLASSID,5);
  PetscValidHeaderSpecific(F,VEC_CLASSID,6);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,7);

  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (iga->form->ops->I2Function) {
    ierr = IGAComputeI2Function(iga,a,A,v,V,t,U,F);CHKERRQ(ierr);
  } else {
    ierr = IGAComputeIFunction(iga,a,A,t,U,F);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGATSFormI2Jacobian(TS ts,PetscReal t,Vec U,Vec V,Vec A,PetscReal shiftV,PetscReal shiftA,Mat J,Mat P,void *ctx)
{
  IGA            iga = (IGA)ctx;
  PetscReal      a=shiftA,v=shiftV;
  PetscReal      dt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(A,VEC_CLASSID,5);
  PetscValidHeaderSpecific(J,MAT_CLASSID,8);
  PetscValidHeaderSpecific(P,MAT_CLASSID,9);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,10);

  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (iga->form->ops->I2Function) {
    ierr = IGAComputeI2Jacobian(iga,a,A,v,V,t,U,P);CHKERRQ(ierr);
  } else {
    ierr = IGAComputeIJacobian(iga,a,A,t,U,P);CHKERRQ(ierr);
  }
  if (J != P) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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
PetscErrorCode IGACreateTS2(IGA iga,TS *ts)
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
  ierr = PetscObjectCompose((PetscObject)*ts,"IGA",(PetscObject)iga);CHKERRQ(ierr);
  ierr = IGASetOptionsHandlerTS(*ts);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&V);CHKERRQ(ierr);
  ierr = TS2SetSolution(*ts,U,V);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&V);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&F);CHKERRQ(ierr);
  ierr = TSSetIFunction (*ts,F,IGATSFormIFunction,iga);CHKERRQ(ierr);
  ierr = TSSetI2Function(*ts,F,IGATSFormI2Function,iga);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);

  ierr = IGACreateMat(iga,&J);CHKERRQ(ierr);
  ierr = TSSetIJacobian (*ts,J,J,IGATSFormIJacobian,iga);CHKERRQ(ierr);
  ierr = TSSetI2Jacobian(*ts,J,J,IGATSFormI2Jacobian,iga);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
