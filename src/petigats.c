#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "IGAFormIFunction"
PetscErrorCode IGAFormIFunction(IGA iga,PetscReal dt,PetscReal shift,
                                PetscReal t,Vec vecU,Vec vecV,Vec vecF,
                                IGAUserIFunction IFunction, void *ctx)
{
  Vec               localU;
  Vec               localV;
  const PetscScalar *arrayU;
  const PetscScalar *arrayV;
  IGAElement        element;
  IGAPoint          point;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,2);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,3);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,4);

  /* Clear global vector F */
  ierr = VecZeroEntries(vecF);CHKERRQ(ierr);

  /* Get local vectors U and V */
  ierr = IGAGetLocalVec(iga,&localU);CHKERRQ(ierr);
  ierr = IGAGetLocalVec(iga,&localV);CHKERRQ(ierr);
  /* Communicate global to local */
  ierr = IGAGlobalToLocal(iga,vecU,localU);CHKERRQ(ierr);
  ierr = IGAGlobalToLocal(iga,vecV,localV);CHKERRQ(ierr);
  /* Get array from the local vectors */
  ierr = VecGetArrayRead(localU,&arrayU);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localV,&arrayV);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGAGetElement(iga,&element);CHKERRQ(ierr);
  ierr = IGAElementBegin(element);CHKERRQ(ierr);
  while (IGAElementNext(element)) {
    PetscScalar *U, *V, *F;
    ierr = IGAElementGetWorkVec(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&V);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&F);CHKERRQ(ierr);
    /* */
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,V);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementGetPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAPointBegin(point);CHKERRQ(ierr);
    while (IGAPointNext(point)) {
      PetscScalar *R;
      ierr = IGAPointGetWorkVec(point,&R);CHKERRQ(ierr);
      ierr = IFunction(point,dt,shift,t,U,V,R,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddVec(point,R,F);CHKERRQ(ierr);
    }
    /* */
    ierr = IGAElementFixFunction(element,F);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,F,vecF);CHKERRQ(ierr);
  }

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
                                PetscReal t,Vec vecU,Vec vecV,Mat matJ,
                                IGAUserIJacobian IJacobian,void *ctx)
{
  Vec               localU;
  Vec               localV;
  const PetscScalar *arrayU;
  const PetscScalar *arrayV;
  IGAElement        element;
  IGAPoint          point;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,3);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,4);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,6);

  /* Clear global matrix J*/
  ierr = MatZeroEntries(matJ);CHKERRQ(ierr);

  /* Get local vectors U and V */
  ierr = IGAGetLocalVec(iga,&localU);CHKERRQ(ierr);
  ierr = IGAGetLocalVec(iga,&localV);CHKERRQ(ierr);
  /* Communicate global to local */
  ierr = IGAGlobalToLocal(iga,vecU,localU);CHKERRQ(ierr);
  ierr = IGAGlobalToLocal(iga,vecV,localV);CHKERRQ(ierr);
  /* Get array from the local vectors */
  ierr = VecGetArrayRead(localU,&arrayU);CHKERRQ(ierr);
  ierr = VecGetArrayRead(localV,&arrayV);CHKERRQ(ierr);

  /* Element Loop */
  ierr = IGAGetElement(iga,&element);CHKERRQ(ierr);
  ierr = IGAElementBegin(element);CHKERRQ(ierr);
  while (IGAElementNext(element)) {
    PetscScalar *U, *V, *J;
    ierr = IGAElementGetWorkVec(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&V);CHKERRQ(ierr);
    ierr = IGAElementGetWorkMat(element,&J);CHKERRQ(ierr);
    /* */
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,V);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementGetPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAPointBegin(point);CHKERRQ(ierr);
    while (IGAPointNext(point)) {
      PetscScalar *K;
      ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
      ierr = IJacobian(point,dt,shift,t,U,V,K,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddMat(point,K,J);CHKERRQ(ierr);
    }
    /* */
    ierr = IGAElementFixJacobian(element,J);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,J,matJ);CHKERRQ(ierr);
  }

  /* Restore array to the vectors */
  ierr = VecRestoreArrayRead(localU,&arrayU);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(localV,&arrayV);CHKERRQ(ierr);
  /* Restore local vectors U and V */
  ierr = IGARestoreLocalVec(iga,&localV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVec(iga,&localU);CHKERRQ(ierr);

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
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = IGAFormIFunction(iga,dt,shift,t,U,V,F,
                          iga->userops->IFunction,
                          iga->userops->IFunCtx);CHKERRQ(ierr);
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
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = IGAFormIJacobian(iga,dt,shift,t,U,V,*P,
                          iga->userops->IJacobian,
                          iga->userops->IJacCtx);CHKERRQ(ierr);
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
  /*ierr = TSSetDM(*ts,iga->dm_dof);CHKERRQ(ierr);*/
  PetscFunctionReturn(0);
}
