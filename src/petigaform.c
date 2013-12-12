#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "IGAGetForm"
PetscErrorCode IGAGetForm(IGA iga,IGAForm *form)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(form,2);
  *form = iga->form;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetForm"
PetscErrorCode IGASetForm(IGA iga,IGAForm form)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(form,2);
  if (form == iga->form) PetscFunctionReturn(0);
  ierr = IGAFormDestroy(&iga->form);CHKERRQ(ierr);
  iga->form = form;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormCreate"
PetscErrorCode IGAFormCreate(IGAForm *_form)
{
  PetscInt       a,s;
  IGAForm        form;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_form,1);
  ierr = PetscCalloc1(1,_form);CHKERRQ(ierr);
  (*_form)->refct = 1; form = *_form;
  /* */
  form->dof = -1;
  ierr = PetscCalloc1(1,&form->ops);CHKERRQ(ierr);
  for (a=0; a<3; a++)
    for (s=0; s<2; s++) {
      ierr = PetscCalloc1(1,&form->value[a][s]);CHKERRQ(ierr);
      ierr = PetscCalloc1(1,&form->load [a][s]);CHKERRQ(ierr);
    }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormDestroy"
PetscErrorCode IGAFormDestroy(IGAForm *_form)
{
  PetscInt       a,s;
  IGAForm        form;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_form,1);
  form = *_form; *_form = 0;
  if (!form) PetscFunctionReturn(0);
  if (--form->refct > 0) PetscFunctionReturn(0);
  /* */
  ierr = PetscFree(form->ops);CHKERRQ(ierr);
  for (a=0; a<3; a++)
    for (s=0; s<2; s++) {
      ierr = PetscFree(form->value[a][s]);CHKERRQ(ierr);
      ierr = PetscFree(form->load [a][s]);CHKERRQ(ierr);
    }
  ierr = PetscFree(form);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormReset"
PetscErrorCode IGAFormReset(IGAForm form)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!form) PetscFunctionReturn(0);
  PetscValidPointer(form,1);
  form->dof = 0;
  ierr = PetscMemzero(form->ops,sizeof(struct _IGAFormOps));CHKERRQ(ierr);
  ierr = PetscMemzero(form->value,3*2*sizeof(struct _IGAFormBC));CHKERRQ(ierr);
  ierr = PetscMemzero(form->load, 3*2*sizeof(struct _IGAFormBC));CHKERRQ(ierr);
  ierr = PetscMemzero(form->visit,3*2*sizeof(PetscBool));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormReference"
PetscErrorCode IGAFormReference(IGAForm form)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->refct++;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------- */

#define IGAFormUpdateDof(form,field) \
  do { \
    (form)->dof = PetscMax((form)->dof,(field)+1);      \
  } while(0)

#define IGAFormCheckArg(arg,m) \
do { \
  if (arg<0)  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,#arg" must be nonnegative, got %D",arg); \
  if (arg>=m) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,#arg" must be less than %D, got %D",m,arg); \
 } while(0)

PETSC_STATIC_INLINE
void IGAFormBCSetEntry(IGAFormBC bc,PetscInt field,PetscScalar value)
{
  PetscInt k;
  for (k=0; k<bc->count; k++)
    if (bc->field[k] == field) break;
  if (k == bc->count) bc->count++;
  bc->field[k] = field;
  bc->value[k] = value;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSetBoundaryValue"
PetscErrorCode IGAFormSetBoundaryValue(IGAForm form,PetscInt axis,PetscInt side,PetscInt field,PetscScalar value)
{
  PetscFunctionBegin;
  IGAFormCheckArg(axis,3);
  IGAFormCheckArg(side,2);
  IGAFormCheckArg(field,64);
  IGAFormUpdateDof(form,field);
  IGAFormBCSetEntry(form->value[axis][side],field,value);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSetBoundaryLoad"
PetscErrorCode IGAFormSetBoundaryLoad(IGAForm form,PetscInt axis,PetscInt side,PetscInt field,PetscScalar value)
{
  PetscFunctionBegin;
  IGAFormCheckArg(axis,3);
  IGAFormCheckArg(side,2);
  IGAFormCheckArg(field,64);
  IGAFormUpdateDof(form,field);
  IGAFormBCSetEntry(form->load[axis][side],field,value);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSetBoundaryForm"
PetscErrorCode IGAFormSetBoundaryForm(IGAForm form,PetscInt axis,PetscInt side,PetscBool flag)
{
  PetscFunctionBegin;
  IGAFormCheckArg(axis,3);
  IGAFormCheckArg(side,2);
  form->visit[axis][side] = flag ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormClearBoundary"
PetscErrorCode IGAFormClearBoundary(IGAForm form,PetscInt axis,PetscInt side)
{
  PetscFunctionBegin;
  IGAFormCheckArg(axis,3);
  IGAFormCheckArg(side,2);
  form->value[axis][side]->count = 0;
  form->load [axis][side]->count = 0;
  form->visit[axis][side] = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSetVector"
PetscErrorCode IGAFormSetVector(IGAForm form,IGAFormVector Vector,void *VecCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->Vector = Vector;
  form->ops->VecCtx = VecCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSetMatrix"
PetscErrorCode IGAFormSetMatrix(IGAForm form,IGAFormMatrix Matrix,void *MatCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->Matrix = Matrix;
  form->ops->MatCtx = MatCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSetSystem"
PetscErrorCode IGAFormSetSystem(IGAForm form,IGAFormSystem System,void *SysCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->System = System;
  form->ops->SysCtx = SysCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSetFunction"
PetscErrorCode IGAFormSetFunction(IGAForm form,IGAFormFunction Function,void *FunCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->Function = Function;
  form->ops->FunCtx   = FunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSetJacobian"
PetscErrorCode IGAFormSetJacobian(IGAForm form,IGAFormJacobian Jacobian,void *JacCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->Jacobian = Jacobian;
  form->ops->JacCtx   = JacCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSetIFunction"
PetscErrorCode IGAFormSetIFunction(IGAForm form,IGAFormIFunction IFunction,void *IFunCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->IFunction = IFunction;
  form->ops->IFunCtx   = IFunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSetIJacobian"
PetscErrorCode IGAFormSetIJacobian(IGAForm form,IGAFormIJacobian IJacobian,void *IJacCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->IJacobian = IJacobian;
  form->ops->IJacCtx   = IJacCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSetIFunction2"
PetscErrorCode IGAFormSetIFunction2(IGAForm form,IGAFormIFunction2 IFunction,void *IFunCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->IFunction2 = IFunction;
  form->ops->IFunCtx    = IFunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSetIJacobian2"
PetscErrorCode IGAFormSetIJacobian2(IGAForm form,IGAFormIJacobian2 IJacobian,void *IJacCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->IJacobian2 = IJacobian;
  form->ops->IJacCtx    = IJacCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSetIEFunction"
PetscErrorCode IGAFormSetIEFunction(IGAForm form,IGAFormIEFunction IEFunction,void *IEFunCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->IEFunction = IEFunction;
  form->ops->IEFunCtx   = IEFunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFormSetIEJacobian"
PetscErrorCode IGAFormSetIEJacobian(IGAForm form,IGAFormIEJacobian IEJacobian,void *IEJacCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->IEJacobian = IEJacobian;
  form->ops->IEJacCtx   = IEJacCtx;
  PetscFunctionReturn(0);
}


/* --------------------------------------------------------------- */

#undef  __FUNCT__
#define __FUNCT__ "IGASetBoundaryValue"
PetscErrorCode IGASetBoundaryValue(IGA iga,PetscInt axis,PetscInt side,PetscInt field,PetscScalar value)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (iga->dim > 0) IGAFormCheckArg(axis,iga->dim);
  if (iga->dof > 0) IGAFormCheckArg(field,iga->dof);
  ierr = IGAFormSetBoundaryValue(iga->form,axis,side,field,value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetBoundaryLoad"
PetscErrorCode IGASetBoundaryLoad(IGA iga,PetscInt axis,PetscInt side,PetscInt field,PetscScalar value)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (iga->dim > 0) IGAFormCheckArg(axis,iga->dim);
  if (iga->dof > 0) IGAFormCheckArg(field,iga->dof);
  ierr = IGAFormSetBoundaryLoad(iga->form,axis,side,field,value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetBoundaryForm"
PetscErrorCode IGASetBoundaryForm(IGA iga,PetscInt axis,PetscInt side,PetscBool flag)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (iga->dim > 0) IGAFormCheckArg(axis,iga->dim);
  ierr = IGAFormSetBoundaryForm(iga->form,axis,side,flag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetFormVector"
/*@
   IGASetFormSystem - Set the user callback to form the vector
   which represents the discretized L(w).

   Logically collective on IGA

   Input Parameters:
+  iga - the IGA context
.  Vector - the function which evaluates L(w)
-  VecCtx - user-defined context for evaluation routine (may be NULL)

   Details of Vector:
$  PetscErrorCode Vector(IGAPoint p,PetscScalar *F,void *ctx);

+  p - point at which to evaluate L(w)
.  F - contribution to L(w)
-  ctx - user-defined context for evaluation routine

   Level: normal

.keywords: IGA, setup linear system, vector assembly
@*/
PetscErrorCode IGASetFormVector(IGA iga,IGAFormVector Vector,void *VecCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->Vector = Vector;
  iga->form->ops->VecCtx = VecCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetFormMatrix"
/*@
   IGASetFormSystem - Set the user callback to form the matrix and vector
   which represents the discretized a(w,u).

   Logically collective on IGA

   Input Parameters:
+  iga - the IGA context
.  Matrix - the function which evaluates a(w,u)
-  MatCtx - user-defined context for evaluation routine (may be NULL)

   Details of System:
$  PetscErrorCode System(IGAPoint p,PetscScalar *K,void *ctx);

+  p - point at which to evaluate a(w,u)
.  K - contribution to a(w,u)
-  ctx - user-defined context for evaluation routine

   Level: normal

.keywords: IGA, setup linear system, matrix assembly
@*/
PetscErrorCode IGASetFormMatrix(IGA iga,IGAFormMatrix Matrix,void *MatCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->Matrix = Matrix;
  iga->form->ops->MatCtx = MatCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetFormSystem"
/*@
   IGASetFormSystem - Set the user callback to form the matrix and vector
   which represents the discretized a(w,u) = L(w).

   Logically collective on IGA

   Input Parameters:
+  iga - the IGA context
.  System - the function which evaluates a(w,u) and L(w)
-  SysCtx - user-defined context for evaluation routine (may be NULL)

   Details of System:
$  PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx);

+  p - point at which to evaluate a(w,u)=L(w)
.  K - contribution to a(w,u)
.  F - contribution to L(w)
-  ctx - user-defined context for evaluation routine

   Level: normal

.keywords: IGA, setup linear system, matrix assembly, vector assembly
@*/
PetscErrorCode IGASetFormSystem(IGA iga,IGAFormSystem System,void *SysCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->System = System;
  iga->form->ops->SysCtx = SysCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetFormFunction"
/*@
   IGASetFormFunction - Set the function which computes the residual vector
   F(U)=0 for use with nonlinear problems.

   Logically Collective on IGA

   Input Parameter:
+  iga - the IGA context
.  Function - the function evaluation routine
-  FunCtx - user-defined context for private data for the function evaluation routine (may be NULL)

   Details of Function:
$  PetscErrorCode Function(IGAPoint p,const PetscScalar *U,PetscScalar *R,void *ctx);

+  p - point at which to compute the residual
.  U - local state vector
.  R - local contribution to global residual vector
-  ctx - [optional] user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetFormFunction(IGA iga,IGAFormFunction Function,void *FunCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->Function = Function;
  iga->form->ops->FunCtx   = FunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetFormJacobian"
/*@
   IGASetFormIJacobian - Set the function to compute the Jacobian matrix
   J = dF/dU where F(U) is the residual function you provided with IGASetFormIFunction().

   Logically Collective on IGA

   Input Parameter:
+  iga - the IGA context
.  Jacobian - the Jacobian evaluation routine
-  JacCtx - user-defined context for private data for the Jacobian evaluation routine (may be NULL)

   Details of Jacobian:
$  PetscErrorCode Jacobian(IGAPoint p,const PetscScalar *U,PetscScalar *J,void *ctx);

+  p - point at which to compute the Jacobian
.  U - local state vector
.  J - local contribution to global Jacobian matrix
-  ctx - [optional] user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetFormJacobian(IGA iga,IGAFormJacobian Jacobian,void *JacCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->Jacobian = Jacobian;
  iga->form->ops->JacCtx   = JacCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetFormIFunction"
/*@
   IGASetFormIFunction - Set the function which computes the residual vector
   F(t,U_t,U)=0 for use with implicit time stepping routines.

   Logically Collective on IGA

   Input Parameter:
+  iga - the IGA context
.  IFunction - the function evaluation routine
-  IFunCtx - user-defined context for private data for the function evaluation routine (may be NULL)

   Details of IFunction:
$  PetscErrorCode IFunction(IGAPoint p,PetscReal dt,
                            PetscReal a,const PetscScalar *V,
                            PetscReal t,const PetscScalar *U,
                            PetscScalar *R,void *ctx);

+  p - point at which to compute the residual
.  dt - time step size
.  a - positive parameter which depends on the time integration method (XXX Should this be here?)
.  V - time derivative of the state vector
.  t - time at step/stage being solved
.  U - state vector
.  R - function vector
-  ctx - [optional] user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetFormIFunction(IGA iga,IGAFormIFunction IFunction,void *IFunCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->IFunction = IFunction;
  iga->form->ops->IFunCtx   = IFunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetFormIJacobian"
/*@
   IGASetFormIJacobian - Set the function to compute the Jacobian matrix
   J = a*dF/dU_t + dF/dU where F(t,U_t,U) is the residual function
   you provided with IGASetFormIFunction().

   Logically Collective on IGA

   Input Parameter:
+  iga - the IGA context
.  IJacobian - the Jacobian evaluation routine
-  IJacCtx - user-defined context for private data for the Jacobian evaluation routine (may be NULL)

   Details of IJacobian:
$  PetscErrorCode IJacobian(IGAPoint p,PetscReal dt,
                            PetscReal a,const PetscScalar *V,
                            PetscReal t,const PetscScalar *U,
                            PetscScalar *J,void *ctx);

+  p - point at which to compute the Jacobian
.  dt - time step size
.  a - positive parameter which depends on the time integration method
.  V - time derivative of the state vector
.  t - time at step/stage being solved
.  U - state vector
.  J - Jacobian matrix
-  ctx - [optional] user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetFormIJacobian(IGA iga,IGAFormIJacobian IJacobian,void *IJacCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->IJacobian = IJacobian;
  iga->form->ops->IJacCtx   = IJacCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetFormIFunction2"
/*@
   IGASetFormIFunction - Set the function which computes the residual vector
   F(t,U_tt,U_t,U)=0 for use with implicit time stepping routines.

   Logically Collective on IGA

   Input Parameter:
+  iga - the IGA context
.  IFunction - the function evaluation routine
-  IFunCtx - user-defined context for private data for the function evaluation routine (may be NULL)

   Details of IFunction:
$  PetscErrorCode IFunction(IGAPoint p,PetscReal dt,
                            PetscReal a,const PetscScalar *A,
                            PetscReal v,const PetscScalar *V,
                            PetscReal t,const PetscScalar *U,
                            PetscScalar *F,void *ctx);

+  p - point at which to compute the residual
.  dt - time step size
.  a - positive parameter which depends on the time integration method
.  A - second time derivative of the state vector
.  v - positive parameter which depends on the time integration method
.  V - time derivative of the state vector
.  t - time at step/stage being solved
.  U - state vector
.  F - function vector
-  ctx - [optional] user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetFormIFunction2(IGA iga,IGAFormIFunction2 IFunction,void *IFunCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->IFunction2 = IFunction;
  iga->form->ops->IFunCtx    = IFunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetFormIJacobian2"
/*@
   IGASetFormIJacobian2 - Set the function to compute the Jacobian matrix
   J = a*dF/dU_tt + v*dF/dU_t + dF/dU  where F(t,U_tt,U_t,U) is
   the function you provided with IGASetFormIFunction2().

   Logically Collective on IGA

   Input Parameter:
+  iga       - the IGA context
.  IJacobian - the Jacobian evaluation routine
-  IJacCtx   - user-defined context for private data for the Jacobian evaluation routine (may be NULL)

   Details of IJacobian:
$  PetscErrorCode IJacobian(IGAPoint p,PetscReal dt,
                            PetscReal a,const PetscScalar *A,
                            PetscReal v,const PetscScalar *V,
                            PetscReal t,const PetscScalar *U,
                            PetscScalar *J,void *ctx);

+  p   - point at which to compute the Jacobian
.  dt  - time step size
.  a   - positive parameter which depends on the time integration method
.  A   - second time derivative of the state vector
.  v   - positive parameter which depends on the time integration method
.  V   - time derivative of the state vector
.  t   - time at step/stage being solved
.  U   - state vector
.  J   - Jacobian matrix
-  ctx - [optional] user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetFormIJacobian2(IGA iga,IGAFormIJacobian2 IJacobian,void *IJacCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->IJacobian2 = IJacobian;
  iga->form->ops->IJacCtx    = IJacCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetFormIEFunction"
/*@
   IGASetFormIEFunction - Set the function which computes the residual vector
   F(t,U_t,U,U0)=0 for use with explicit or implicit time stepping routines.

   Logically Collective on IGA

   Input Parameter:
+  iga - the IGA context
.  IEFunction - the function evaluation routine
-  IEFunCtx - user-defined context for private data for the function evaluation routine (may be NULL)

   Details of IEFunction:
$  PetscErrorCode IEFunction(IGAPoint p,PetscReal dt,
                             PetscReal a, const PetscScalar *V,
                             PetscReal t, const PetscScalar *U,
                             PetscReal t0,const PetscScalar *U0,
                             PetscScalar *R,void *ctx);

+  p - point at which to compute the residual
.  dt - time step size
.  a - positive parameter which depends on the time integration method (XXX Should this be here?)
.  V - time derivative of the state vector at t0
.  t - time at step/stage being solved
.  U - state vector at t
.  t0 - time at current step
.  U0 - state vector at t0
.  R - function vector
-  ctx - [optional] user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetFormIEFunction(IGA iga,IGAFormIEFunction IEFunction,void *IEFunCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->IEFunction = IEFunction;
  iga->form->ops->IEFunCtx   = IEFunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetFormIEJacobian"
/*@
   IGASetFormIEJacobian - Set the function to compute the Jacobian matrix
   J = a*dF/dU_t + dF/dU where F(t,U_t,U,U0) is the function you provided with
   IGASetFormIEFunction(). For use with implicit+explicit TS methods.

   Logically Collective on IGA

   Input Parameter:
+  iga - the IGA context
.  IEJacobian - the Jacobian evaluation routine
-  IEJacCtx - user-defined context for private data for the Jacobian evaluation routine (may be NULL)

   Details of IEJacobian:
$  PetscErrorCode IEJacobian(IGAPoint p,PetscReal dt,
                             PetscReal a, const PetscScalar *V,
                             PetscReal t, const PetscScalar *U,
                             PetscReal t0,const PetscScalar *U0,
                             PetscScalar *J,void *ctx);

+  p - point at which to compute the Jacobian
.  dt - time step size
.  a - positive parameter which depends on the time integration method
.  V - time derivative of the state vector at t0
.  t - time at step/stage being solved
.  U - state vector at t
.  t0 - time at current step
.  U0 - state vector at t0
.  J - Jacobian matrix
-  ctx - [optional] user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetFormIEJacobian(IGA iga,IGAFormIEJacobian IEJacobian,void *IEJacCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->IEJacobian = IEJacobian;
  iga->form->ops->IEJacCtx   = IEJacCtx;
  PetscFunctionReturn(0);
}
