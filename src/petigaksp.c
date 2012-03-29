#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSystem"
PetscErrorCode IGAFormSystem(IGA iga,Mat matA,Vec vecB,IGAUserSystem System,void *ctx)
{
  IGAElement     element;
  IGAPoint       point;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(matA,MAT_CLASSID,2);
  PetscValidHeaderSpecific(vecB,VEC_CLASSID,3);
  IGACheckSetUp(iga,1);

  ierr = MatZeroEntries(matA);CHKERRQ(ierr);
  ierr = VecZeroEntries(vecB);CHKERRQ(ierr);

  ierr = IGAGetElement(iga,&element);CHKERRQ(ierr);
  ierr = IGAElementBegin(element);CHKERRQ(ierr);
  while (IGAElementNext(element)) {
    PetscScalar *A, *B;
    ierr = IGAElementGetWorkMat(element,&A);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&B);CHKERRQ(ierr);
    ierr = IGAElementGetPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAPointBegin(point);CHKERRQ(ierr);
    while (IGAPointNext(point)) {
      PetscScalar *K, *F;
      ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
      ierr = IGAPointGetWorkVec(point,&F);CHKERRQ(ierr);
      ierr = System(point,K,F,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddMat(point,K,A);CHKERRQ(ierr);
      ierr = IGAPointAddVec(point,F,B);CHKERRQ(ierr);
    }
    ierr = IGAElementFixSystem(element,A,B);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,A,matA);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,B,vecB);CHKERRQ(ierr);
  }
  ierr = IGAElementEnd(element);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vecB);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (vecB);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateKSP"
PetscErrorCode IGACreateKSP(IGA iga, KSP *ksp)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(ksp,2);
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = KSPCreate(comm,ksp);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*ksp,"IGA",(PetscObject)iga);CHKERRQ(ierr);
  /*ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);*/
  /*ierr = KSPSetOperators(*ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);*/
  /*ierr = MatDestroy(&A);CHKERRQ(ierr);*/
  /*ierr = KSPSetDM(**ksp,iga->dm_dof);CHKERRQ(ierr);*/
  /*ierr = KSPSetDMActive(**ksp,PETSC_FALSE);CHKERRQ(ierr);*/

  PetscFunctionReturn(0);
}
