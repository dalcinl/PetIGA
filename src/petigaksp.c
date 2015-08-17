#include "petiga.h"

PETSC_STATIC_INLINE
PetscBool IGAElementNextFormVector(IGAElement element,IGAFormVector *vec,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *vec = form->ops->Vector;
  *ctx = form->ops->VecCtx;
  return PETSC_TRUE;
}

PETSC_STATIC_INLINE
PetscBool IGAElementNextFormMatrix(IGAElement element,IGAFormMatrix *mat,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *mat = form->ops->Matrix;
  *ctx = form->ops->MatCtx;
  return PETSC_TRUE;
}

PETSC_STATIC_INLINE
PetscBool IGAElementNextFormSystem(IGAElement element,IGAFormSystem *sys,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *sys = form->ops->System;
  *ctx = form->ops->SysCtx;
  return PETSC_TRUE;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeVector"
PetscErrorCode IGAComputeVector(IGA iga,Vec vecB)
{
  IGAElement     element;
  IGAPoint       point;
  IGAFormVector  Vector;
  void           *ctx;
  PetscScalar    *B;
  PetscScalar    *F;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecB,VEC_CLASSID,2);
  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,Vector);

  ierr = VecZeroEntries(vecB);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormVector,iga,vecB,0,0);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkVec(element,&B);CHKERRQ(ierr);
    /* FormVector loop */
    while (IGAElementNextFormVector(element,&Vector,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkVec(point,&F);CHKERRQ(ierr);
        ierr = Vector(point,F,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddVec(point,F,B);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementAssembleVec(element,B,vecB);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormVector,iga,vecB,0,0);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(vecB);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (vecB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeMatrix"
PetscErrorCode IGAComputeMatrix(IGA iga,Mat matA)
{
  IGAElement     element;
  IGAPoint       point;
  IGAFormMatrix  Matrix;
  void           *ctx;
  PetscScalar    *A;
  PetscScalar    *K;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(matA,MAT_CLASSID,2);
  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,Matrix);

  ierr = MatZeroEntries(matA);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormMatrix,iga,matA,0,0);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkMat(element,&A);CHKERRQ(ierr);
    /* FormMatrix loop */
    while (IGAElementNextFormMatrix(element,&Matrix,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
        ierr = Matrix(point,K,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddMat(point,K,A);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementAssembleMat(element,A,matA);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormMatrix,iga,matA,0,0);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "IGAComputeSystem"
/*@
   IGAComputeSystem - Form the matrix and vector which represents the
   discretized a(w,u) = L(w).

   Collective on IGA/Mat/Vec

   Input Parameters:
.  iga - the IGA context

   Output Parameters:
+  matA - the matrix obtained from discretization of a(w,u)
-  vecB - the vector obtained from discretization of L(w)

   Notes:
   This routine is used to solve a steady, linear problem. It performs
   matrix/vector assembly standard in FEM. The form provides a routine
   which evaluates the bilinear and linear forms at a point.

   Level: normal

.keywords: IGA, setup linear system, matrix assembly, vector assembly
@*/
PetscErrorCode IGAComputeSystem(IGA iga,Mat matA,Vec vecB)
{
  IGAElement     element;
  IGAPoint       point;
  IGAFormSystem  System;
  void           *ctx;
  PetscScalar    *A,*B;
  PetscScalar    *K,*F;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(matA,MAT_CLASSID,2);
  PetscValidHeaderSpecific(vecB,VEC_CLASSID,3);
  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,System);

  ierr = MatZeroEntries(matA);CHKERRQ(ierr);
  ierr = VecZeroEntries(vecB);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormSystem,iga,matA,vecB,0);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetWorkMat(element,&A);CHKERRQ(ierr);
    ierr = IGAElementGetWorkVec(element,&B);CHKERRQ(ierr);
    /* FormSystem loop */
    while (IGAElementNextFormSystem(element,&System,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point)) {
        ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
        ierr = IGAPointGetWorkVec(point,&F);CHKERRQ(ierr);
        ierr = System(point,K,F,ctx);CHKERRQ(ierr);
        ierr = IGAPointAddMat(point,K,A);CHKERRQ(ierr);
        ierr = IGAPointAddVec(point,F,B);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementFixSystem(element,A,B);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,A,matA);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,B,vecB);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormSystem,iga,matA,vecB,0);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vecB);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (vecB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAKSPFormRHS"
PetscErrorCode IGAKSPFormRHS(KSP ksp,Vec b,void *ctx)
{
  IGA            iga = (IGA)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,3);
  if (!iga->form->ops->System) {
    ierr = IGAComputeVector(iga,b);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAKSPFormOperators"
PetscErrorCode IGAKSPFormOperators(KSP ksp,Mat A,Mat B,void *ctx)
{
  IGA            iga = (IGA)ctx;
  Vec            rhs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,4);
  if (!iga->form->ops->System) {
    ierr = IGAComputeMatrix(iga,A);CHKERRQ(ierr);
  } else {
    ierr = KSPGetRhs(ksp,&rhs);CHKERRQ(ierr);
    ierr = IGAComputeSystem(iga,A,rhs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
#undef  __FUNCT__
#define __FUNCT__ "IGA_OptionsHandler_KSP"
static PetscErrorCode IGA_OptionsHandler_KSP(PetscObject obj,void *ctx)
{
  KSP            ksp = (KSP)obj;
  IGA            iga;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (PetscOptionsPublishCount != 1) PetscFunctionReturn(0);
  ierr = PetscObjectQuery((PetscObject)ksp,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  if (!iga) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscFunctionReturn(0);
}
*/

#if PETSC_VERSION_LT(3,5,0)
#define PCGetOperators(pc,A,B) PCGetOperators(pc,A,B,NULL)
#endif

#undef  __FUNCT__
#define __FUNCT__ "IGA_OptionsHandler_PC"
static PetscErrorCode IGA_OptionsHandler_PC(PetscObject obj,PETSC_UNUSED void *ctx)
{
  PC             pc = (PC)obj;
  Mat            mat;
  IGA            iga;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
#if PETSC_VERSION_LT(3,6,0)
  if (PetscOptionsPublishCount != 1) PetscFunctionReturn(0);
#endif
  ierr = PCGetOperators(pc,NULL,&mat);CHKERRQ(ierr);
  if (!mat) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)mat,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  if (!iga) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  /* */
  ierr = IGAPreparePCMG(iga,pc);CHKERRQ(ierr);
  ierr = IGAPreparePCBDDC(iga,pc);CHKERRQ(ierr);
  /* */
  PetscFunctionReturn(0);
}

static PetscErrorCode OptHdlDel(PETSC_UNUSED PetscObject obj,PETSC_UNUSED void *ctx) {return 0;}

#undef  __FUNCT__
#define __FUNCT__ "IGASetOptionsHandlerKSP"
PetscErrorCode IGASetOptionsHandlerKSP(KSP ksp)
{
  PC             pc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  /*ierr = PetscObjectAddOptionsHandler((PetscObject)ksp,IGA_OptionsHandler_KSP,OptHdlDel,NULL);CHKERRQ(ierr);*/
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = IGASetOptionsHandlerPC(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetOptionsHandlerPC"
PetscErrorCode IGASetOptionsHandlerPC(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscObjectAddOptionsHandler((PetscObject)pc,IGA_OptionsHandler_PC,OptHdlDel,NULL);CHKERRQ(ierr);
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
PetscErrorCode IGACreateKSP(IGA iga,KSP *ksp)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(ksp,2);

  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = KSPCreate(comm,ksp);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*ksp,"IGA",(PetscObject)iga);CHKERRQ(ierr);
  ierr = IGASetOptionsHandlerKSP(*ksp);CHKERRQ(ierr);

  /*ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);*/
  /*ierr = KSPSetOperators(*ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);*/
  /*ierr = MatDestroy(&A);CHKERRQ(ierr);*/
  PetscFunctionReturn(0);
}
