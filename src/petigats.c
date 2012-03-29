#include "petiga.h"

extern PetscLogEvent IGA_FormFunction;
extern PetscLogEvent IGA_FormJacobian;

#undef  __FUNCT__
#define __FUNCT__ "IGAFormIFunction"
PetscErrorCode IGAFormIFunction(IGA iga,PetscReal dt,PetscReal shift,
                                PetscReal t,Vec vecV,Vec vecU,
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
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,5);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,7);
  IGACheckSetUp(iga,1);

  /* Clear global vector F */
  ierr = VecZeroEntries(vecF);CHKERRQ(ierr);

  /* Get local vectors U and V */
  ierr = IGAGetLocalVec(iga,&localV);CHKERRQ(ierr);
  ierr = IGAGetLocalVec(iga,&localU);CHKERRQ(ierr);
  /* Communicate global to local */
  ierr = IGAGlobalToLocal(iga,vecV,localV);CHKERRQ(ierr);
  ierr = IGAGlobalToLocal(iga,vecU,localU);CHKERRQ(ierr);
  /* Get array from the local vectors */
  ierr = VecGetArrayRead(localV,&arrayV);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localU,&arrayU);CHKERRQ(ierr);

  /* Element loop */
  ierr = PetscLogEventBegin(IGA_FormFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);
  ierr = IGAGetElement(iga,&element);CHKERRQ(ierr);
  ierr = IGAElementBegin(element);CHKERRQ(ierr);
  while (IGAElementNext(element)) {
    PetscScalar *V, *U, *F;
    ierr = IGAElementGetWorkVec(element,&V);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&F);CHKERRQ(ierr);
    /* */
    ierr = IGAElementGetValues(element,arrayV,V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementGetPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAPointBegin(point);CHKERRQ(ierr);
    while (IGAPointNext(point)) {
      PetscScalar *R;
      ierr = IGAPointGetWorkVec(point,&R);CHKERRQ(ierr);
      ierr = IFunction(point,dt,shift,t,V,U,R,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddVec(point,R,F);CHKERRQ(ierr);
    }
    /* */
    ierr = IGAElementFixFunction(element,F);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,F,vecF);CHKERRQ(ierr);
  }
  ierr = IGAElementEnd(element);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IGA_FormFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);

  /* Restore array to the local vectors */
  ierr = VecRestoreArrayRead(localV,&arrayV);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(localU,&arrayU);CHKERRQ(ierr);
  /* Restore local vectors U and V */
  ierr = IGARestoreLocalVec(iga,&localV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVec(iga,&localU);CHKERRQ(ierr);

  /* Assemble global vector F */
  ierr = VecAssemblyBegin(vecF);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecF);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormIJacobian"
PetscErrorCode IGAFormIJacobian(IGA iga,PetscReal dt,PetscReal shift,
                                PetscReal t,Vec vecV,Vec vecU,
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

  /* Get local vectors U and V */
  ierr = IGAGetLocalVec(iga,&localV);CHKERRQ(ierr);
  ierr = IGAGetLocalVec(iga,&localU);CHKERRQ(ierr);
  /* Communicate global to local */
  ierr = IGAGlobalToLocal(iga,vecV,localV);CHKERRQ(ierr);
  ierr = IGAGlobalToLocal(iga,vecU,localU);CHKERRQ(ierr);
  /* Get array from the local vectors */
  ierr = VecGetArrayRead(localV,&arrayV);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localU,&arrayU);CHKERRQ(ierr);

  /* Element Loop */
  ierr = PetscLogEventBegin(IGA_FormJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);
  ierr = IGAGetElement(iga,&element);CHKERRQ(ierr);
  ierr = IGAElementBegin(element);CHKERRQ(ierr);
  while (IGAElementNext(element)) {
    PetscScalar *V, *U, *J;
    ierr = IGAElementGetWorkVec(element,&V);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkMat(element,&J);CHKERRQ(ierr);
    /* */
    ierr = IGAElementGetValues(element,arrayV,V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementGetPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAPointBegin(point);CHKERRQ(ierr);
    while (IGAPointNext(point)) {
      PetscScalar *K;
      ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
      ierr = IJacobian(point,dt,shift,t,V,U,K,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddMat(point,K,J);CHKERRQ(ierr);
    }
    /* */
    ierr = IGAElementFixJacobian(element,J);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,J,matJ);CHKERRQ(ierr);
  }
  ierr = IGAElementEnd(element);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IGA_FormJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);

  /* Restore array to the vectors */
  ierr = VecRestoreArrayRead(localV,&arrayV);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(localU,&arrayU);CHKERRQ(ierr);
  /* Restore local vectors U and V */
  ierr = IGARestoreLocalVec(iga,&localV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVec(iga,&localU);CHKERRQ(ierr);

  /* Assemble global matrix J*/
  ierr = MatAssemblyBegin(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormIEFunction"
PetscErrorCode IGAFormIEFunction(IGA iga,PetscReal dt,PetscReal shift,
                                 PetscReal t,Vec vecV,Vec vecU,Vec vecU0,
                                 Vec vecF,
                                 IGAUserIEFunction IEFunction, void *ctx)
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
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,8);
  IGACheckSetUp(iga,1);

  /* Clear global vector F */
  ierr = VecZeroEntries(vecF);CHKERRQ(ierr);

  /* Get local vector and array V */
  ierr = IGAGetLocalVec(iga,&localV);CHKERRQ(ierr);
  ierr = IGAGlobalToLocal(iga,vecV,localV);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localV,&arrayV);CHKERRQ(ierr);
  /* Get local vector and array U */
  ierr = IGAGetLocalVec(iga,&localU);CHKERRQ(ierr);
  ierr = IGAGlobalToLocal(iga,vecU,localU);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localU,&arrayU);CHKERRQ(ierr);
  /* Get local vector and array U0 */
  ierr = IGAGetLocalVec(iga,&localU0);CHKERRQ(ierr);
  ierr = IGAGlobalToLocal(iga,vecU0,localU0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localU0,&arrayU0);CHKERRQ(ierr);

  /* Element loop */
  ierr = PetscLogEventBegin(IGA_FormFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);
  ierr = IGAGetElement(iga,&element);CHKERRQ(ierr);
  ierr = IGAElementBegin(element);CHKERRQ(ierr);
  while (IGAElementNext(element)) {
    PetscScalar *V, *U, *U0, *F;
    ierr = IGAElementGetWorkVec(element,&V);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&U0);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&F);CHKERRQ(ierr);
    /* */
    ierr = IGAElementGetValues(element,arrayV,V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU0,U0);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U0);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementGetPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAPointBegin(point);CHKERRQ(ierr);
    while (IGAPointNext(point)) {
      PetscScalar *R;
      ierr = IGAPointGetWorkVec(point,&R);CHKERRQ(ierr);
      ierr = IEFunction(point,dt,shift,t,V,U,U0,R,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddVec(point,R,F);CHKERRQ(ierr);
    }
    /* */
    ierr = IGAElementFixFunction(element,F);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,F,vecF);CHKERRQ(ierr);
  }
  ierr = IGAElementEnd(element);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IGA_FormFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);

  /* Restore array and local vector V */
  ierr = VecRestoreArrayRead(localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVec(iga,&localV);CHKERRQ(ierr);
  /* Restore array and local vector U */
  ierr = VecRestoreArrayRead(localU,&arrayU);CHKERRQ(ierr);
  ierr = IGARestoreLocalVec(iga,&localU);CHKERRQ(ierr);
  /* Restore array and local vector U0 */
  ierr = VecRestoreArrayRead(localU0,&arrayU0);CHKERRQ(ierr);
  ierr = IGARestoreLocalVec(iga,&localU0);CHKERRQ(ierr);

  /* Assemble global vector F */
  ierr = VecAssemblyBegin(vecF);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecF);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormIEJacobian"
PetscErrorCode IGAFormIEJacobian(IGA iga,PetscReal dt,PetscReal shift,
                                 PetscReal t,Vec vecV,Vec vecU,Vec vecU0,
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

  /* Get local vector and array V */
  ierr = IGAGetLocalVec(iga,&localV);CHKERRQ(ierr);
  ierr = IGAGlobalToLocal(iga,vecV,localV);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localV,&arrayV);CHKERRQ(ierr);
  /* Get local vector and array U */
  ierr = IGAGetLocalVec(iga,&localU);CHKERRQ(ierr);
  ierr = IGAGlobalToLocal(iga,vecU,localU);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localU,&arrayU);CHKERRQ(ierr);
  /* Get local vector and array U0 */
  ierr = IGAGetLocalVec(iga,&localU0);CHKERRQ(ierr);
  ierr = IGAGlobalToLocal(iga,vecU0,localU0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localU0,&arrayU0);CHKERRQ(ierr);

  /* Element Loop */
  ierr = PetscLogEventBegin(IGA_FormJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);
  ierr = IGAGetElement(iga,&element);CHKERRQ(ierr);
  ierr = IGAElementBegin(element);CHKERRQ(ierr);
  while (IGAElementNext(element)) {
    PetscScalar *V, *U, *U0, *J;
    ierr = IGAElementGetWorkVec(element,&V);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&U0);CHKERRQ(ierr);
    ierr = IGAElementGetWorkMat(element,&J);CHKERRQ(ierr);
    /* */
    ierr = IGAElementGetValues(element,arrayV,V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU0,U0);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U0);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementGetPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAPointBegin(point);CHKERRQ(ierr);
    while (IGAPointNext(point)) {
      PetscScalar *K;
      ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
      ierr = IEJacobian(point,dt,shift,t,V,U,U0,K,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddMat(point,K,J);CHKERRQ(ierr);
    }
    /* */
    ierr = IGAElementFixJacobian(element,J);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,J,matJ);CHKERRQ(ierr);
  }
  ierr = IGAElementEnd(element);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IGA_FormJacobian,iga,vecV,vecU,matJ);CHKERRQ(ierr);

  /* Restore array and local vector V */
  ierr = VecRestoreArrayRead(localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVec(iga,&localV);CHKERRQ(ierr);
  /* Restore array and local vector U */
  ierr = VecRestoreArrayRead(localU,&arrayU);CHKERRQ(ierr);
  ierr = IGARestoreLocalVec(iga,&localU);CHKERRQ(ierr);
  /* Restore array and local vector U0 */
  ierr = VecRestoreArrayRead(localU0,&arrayU0);CHKERRQ(ierr);
  ierr = IGARestoreLocalVec(iga,&localU0);CHKERRQ(ierr);

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
  PetscReal      dt,shift=0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(F,VEC_CLASSID,5);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,6);
  if (!iga->userops->IFunction && !iga->userops->IEFunction)
    SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_USER,"Must call IGASetUserIFunction()");
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (iga->userops->IEFunction) {
    Vec U0;
    ierr = TSGetSolution(ts,&U0);CHKERRQ(ierr);
    ierr = IGAFormIEFunction(iga,dt,shift,t,V,U,U0,F,
                             iga->userops->IEFunction,
                             iga->userops->IEFunCtx);CHKERRQ(ierr);
  } else {
    ierr = IGAFormIFunction(iga,dt,shift,t,V,U,F,
                            iga->userops->IFunction,
                            iga->userops->IFunCtx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGATSFormIJacobian"
PetscErrorCode IGATSFormIJacobian(TS ts,PetscReal t,Vec U,Vec V,PetscReal shift,Mat *J,Mat *P,MatStructure *m,void *ctx)
{
  IGA            iga = (IGA)ctx;
  PetscReal      dt;
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
  if (!iga->userops->IJacobian && !iga->userops->IEJacobian)
    SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_USER,"Must call IGASetUserIJacobian()");
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (iga->userops->IEJacobian) {
    Vec U0;
    ierr = TSGetSolution(ts,&U0);CHKERRQ(ierr);
    ierr = IGAFormIEJacobian(iga,dt,shift,t,V,U,U0,*P,
                             iga->userops->IEJacobian,
                             iga->userops->IEJacCtx);CHKERRQ(ierr);
  } else {
    ierr = IGAFormIJacobian(iga,dt,shift,t,V,U,*P,
                            iga->userops->IJacobian,
                            iga->userops->IJacCtx);CHKERRQ(ierr);
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
