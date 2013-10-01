#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "IGABoundaryCreate"
PetscErrorCode IGABoundaryCreate(IGABoundary *boundary)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  ierr = PetscNew(struct _n_IGABoundary,boundary);CHKERRQ(ierr);
  (*boundary)->refct = 1;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundaryDestroy"
PetscErrorCode IGABoundaryDestroy(IGABoundary *_boundary)
{
  IGABoundary    boundary;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_boundary,1);
  boundary = *_boundary; *_boundary = 0;
  if (!boundary) PetscFunctionReturn(0);
  if (--boundary->refct > 0) PetscFunctionReturn(0);
  ierr = IGABoundaryReset(boundary);CHKERRQ(ierr);
  ierr = PetscFree(boundary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundaryReset"
PetscErrorCode IGABoundaryReset(IGABoundary boundary)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!boundary) PetscFunctionReturn(0);
  PetscValidPointer(boundary,1);
  boundary->dof = 0;
  boundary->count = 0;
  ierr = PetscFree(boundary->field);CHKERRQ(ierr);
  ierr = PetscFree(boundary->value);CHKERRQ(ierr);
  boundary->nload = 0;
  ierr = PetscFree(boundary->iload);CHKERRQ(ierr);
  ierr = PetscFree(boundary->vload);CHKERRQ(ierr);
  ierr = PetscFree(boundary->userops);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundaryReference"
PetscErrorCode IGABoundaryReference(IGABoundary boundary)
{
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  boundary->refct++;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundaryInit"
PetscErrorCode IGABoundaryInit(IGABoundary boundary,PetscInt dof)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  ierr = IGABoundaryReset(boundary);CHKERRQ(ierr);
  boundary->dof = dof;
  boundary->count = 0;
  ierr = PetscMalloc1(dof,PetscInt,   &boundary->field);CHKERRQ(ierr);
  ierr = PetscMalloc1(dof,PetscScalar,&boundary->value);CHKERRQ(ierr);
  boundary->nload = 0;
  ierr = PetscMalloc1(dof,PetscInt,   &boundary->iload);CHKERRQ(ierr);
  ierr = PetscMalloc1(dof,PetscScalar,&boundary->vload);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundaryClear"
PetscErrorCode IGABoundaryClear(IGABoundary boundary)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  boundary->count = 0;
  boundary->nload = 0;
  ierr = PetscFree(boundary->userops);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetValue"
/*@
   IGABoundarySetValue - Used to set a constant Dirichlet boundary
   condition on the given boundary.

   Logically Collective on IGABoundary

   Input Parameters:
+  boundary - the IGAAxis context
.  field - the index of the field on which to enforce the condition
-  value - the value to set

   Level: normal

.keywords: IGA, boundary, Dirichlet
@*/
PetscErrorCode IGABoundarySetValue(IGABoundary boundary,PetscInt field,PetscScalar value)
{
  PetscInt dof;
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  dof = boundary->dof;
  if (field <  0)   SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Field %D must be nonnegative",field);
  if (field >= dof) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Field %D, but dof %D",field,dof);
  { /**/
    PetscInt pos;
    for (pos=0; pos<boundary->count; pos++)
      if (boundary->field[pos] == field) break;
    if (pos==boundary->count) boundary->count++;
    boundary->field[pos] = field;
    boundary->value[pos] = value;
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetLoad"
/*@
   IGABoundarySetLoad - Used to set a constant Neumann boundary
   condition on the given boundary.

   Logically Collective on IGABoundary

   Input Parameters:
+  boundary - the IGAAxis context
.  field - the index of the field on which to enforce the condition
-  value - the value to set

   Level: normal

.keywords: IGA, boundary, Neumann
@*/
PetscErrorCode IGABoundarySetLoad(IGABoundary boundary,PetscInt field,PetscScalar value)
{
  PetscInt dof;
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  dof = boundary->dof;
  if (field <  0)   SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Field %D must be nonnegative",field);
  if (field >= dof) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Field %D, but dof %D",field,dof);
  { /**/
    PetscInt pos;
    for (pos=0; pos<boundary->nload; pos++)
      if (boundary->iload[pos] == field) break;
    if (pos==boundary->nload) boundary->nload++;
    boundary->iload[pos] = field;
    boundary->vload[pos] = value;
  }
  PetscFunctionReturn(0);
}

#define IGABoundaryEnsureUserOps(boundary) do { PetscErrorCode _ierr;                                     \
    if (!(boundary)->userops) {_ierr = PetscNew(struct _IGAUserOps,&(boundary)->userops);CHKERRQ(_ierr);} \
  } while (0)

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetUserVector"
PetscErrorCode IGABoundarySetUserVector(IGABoundary boundary,IGAUserVector Vector,void *VecCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  IGABoundaryEnsureUserOps(boundary);
  if (Vector) boundary->userops->Vector = Vector;
  if (VecCtx) boundary->userops->VecCtx = VecCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetUserMatrix"
PetscErrorCode IGABoundarySetUserMatrix(IGABoundary boundary,IGAUserMatrix Matrix,void *MatCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  IGABoundaryEnsureUserOps(boundary);
  if (Matrix) boundary->userops->Matrix = Matrix;
  if (MatCtx) boundary->userops->MatCtx = MatCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetUserSystem"
/*@
   IGABoundarySetUserSystem - Set the user callback to form the matrix
   and vector which represents the discretized a(w,u) = L(w)
   integrated along the given boundary.

   Logically collective on IGABoundary

   Input Parameters:
+  boundary - the IGABoundary context
.  System - the function which evaluates a(w,u) and L(w)
-  ctx - user-defined context for evaluation routine (may be NULL)

   Details of System:
$  PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx);

+  p - point at which to evaluate a(w,u)=L(w)
.  K - contribution to a(w,u)
.  F - contribution to L(w)
-  ctx - user-defined context for evaluation routine

   Level: normal

.keywords: IGABoundary, setup linear system, matrix assembly, vector assembly
@*/
PetscErrorCode IGABoundarySetUserSystem(IGABoundary boundary,IGAUserSystem System,void *SysCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  IGABoundaryEnsureUserOps(boundary);
  if (System) boundary->userops->System = System;
  if (SysCtx) boundary->userops->SysCtx = SysCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetUserFunction"
PetscErrorCode IGABoundarySetUserFunction(IGABoundary boundary,IGAUserFunction Function,void *FunCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(boundary->userops,0);
  IGABoundaryEnsureUserOps(boundary);
  if (Function) boundary->userops->Function = Function;
  if (FunCtx)   boundary->userops->FunCtx   = FunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetUserJacobian"
PetscErrorCode IGABoundarySetUserJacobian(IGABoundary boundary,IGAUserJacobian Jacobian,void *JacCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  IGABoundaryEnsureUserOps(boundary);
  if (Jacobian) boundary->userops->Jacobian = Jacobian;
  if (JacCtx)   boundary->userops->JacCtx   = JacCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetUserIFunction"
PetscErrorCode IGABoundarySetUserIFunction(IGABoundary boundary,IGAUserIFunction IFunction,void *IFunCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  IGABoundaryEnsureUserOps(boundary);
  if (IFunction) boundary->userops->IFunction = IFunction;
  if (IFunCtx)   boundary->userops->IFunCtx   = IFunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetUserIJacobian"
PetscErrorCode IGABoundarySetUserIJacobian(IGABoundary boundary,IGAUserIJacobian IJacobian,void *IJacCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  IGABoundaryEnsureUserOps(boundary);
  if (IJacobian) boundary->userops->IJacobian = IJacobian;
  if (IJacCtx)   boundary->userops->IJacCtx   = IJacCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetUserIFunction2"
PetscErrorCode IGABoundarySetUserIFunction2(IGABoundary boundary,IGAUserIFunction2 IFunction,void *IFunCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  IGABoundaryEnsureUserOps(boundary);
  if (IFunction) boundary->userops->IFunction2 = IFunction;
  if (IFunCtx)   boundary->userops->IFunCtx    = IFunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetUserIJacobian2"
PetscErrorCode IGABoundarySetUserIJacobian2(IGABoundary boundary,IGAUserIJacobian2 IJacobian,void *IJacCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  IGABoundaryEnsureUserOps(boundary);
  if (IJacobian) boundary->userops->IJacobian2 = IJacobian;
  if (IJacCtx)   boundary->userops->IJacCtx    = IJacCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetUserIEFunction"
PetscErrorCode IGABoundarySetUserIEFunction(IGABoundary boundary,IGAUserIEFunction IEFunction,void *IEFunCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  IGABoundaryEnsureUserOps(boundary);
  if (IEFunction) boundary->userops->IEFunction = IEFunction;
  if (IEFunCtx)   boundary->userops->IEFunCtx   = IEFunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetUserIEJacobian"
PetscErrorCode IGABoundarySetUserIEJacobian(IGABoundary boundary,IGAUserIEJacobian IEJacobian,void *IEJacCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  IGABoundaryEnsureUserOps(boundary);
  if (IEJacobian) boundary->userops->IEJacobian = IEJacobian;
  if (IEJacCtx)   boundary->userops->IEJacCtx   = IEJacCtx;
  PetscFunctionReturn(0);
}
