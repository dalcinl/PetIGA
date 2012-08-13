#include "petiga.h"

extern PetscLogEvent IGA_FormSystem;

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSystem"
/*@
   IGAFormSystem - Form the matrix and vector which represents the
   discretized a(w,u) = L(w).
   
   Collective on IGA/Mat/Vec

   Input Parameters:
.  iga - the IGA context

   Output Parameters:
+  matA - the matrix obtained from discretization of a(w,u)
-  vecB - the vector obtained from discretization of L(w)

   Notes: 
   This routine is used to solve a steady, linear problem. It performs
   matrix/vector assembly standard in FEM. The user provides a routine
   which evaluates the bilinear form at a point.

   Level: normal

.keywords: IGA, setup linear system, matrix assembly, vector assembly
@*/
PetscErrorCode IGAFormSystem(IGA iga,Mat matA,Vec vecB)
{
  IGAElement     element;
  IGAPoint       point;
  IGAUserSystem  System;
  void           *SysCtx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(matA,MAT_CLASSID,2);
  PetscValidHeaderSpecific(vecB,VEC_CLASSID,3);
  IGACheckSetUp(iga,1);
  IGACheckUserOp(iga,1,System);

  System = iga->userops->System;
  SysCtx = iga->userops->SysCtx;

  ierr = MatZeroEntries(matA);CHKERRQ(ierr);
  ierr = VecZeroEntries(vecB);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormSystem,iga,matA,vecB,0);CHKERRQ(ierr);
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
      ierr = System(point,K,F,SysCtx);CHKERRQ(ierr);
      ierr = IGAPointAddMat(point,K,A);CHKERRQ(ierr);
      ierr = IGAPointAddVec(point,F,B);CHKERRQ(ierr);
    }
    ierr = IGAElementFixSystem(element,A,B);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,A,matA);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,B,vecB);CHKERRQ(ierr);
  }
  ierr = IGAElementEnd(element);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IGA_FormSystem,iga,matA,vecB,0);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vecB);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (vecB);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateKSP"
/*@
   IGACreateKSP - Creates a KSP (linear solver) which uses the same
   communicators as the IGA.
   
   Logically collective on IGA

   Input Parameter:
.  iga - the IGA context

   Output Parameter:
.  ksp - the KSP

   Level: normal

.keywords: IGA, create, KSP
@*/
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
  PetscFunctionReturn(0);
}
