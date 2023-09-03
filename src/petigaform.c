#include "petiga.h"

PetscErrorCode IGAGetForm(IGA iga,IGAForm *form)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(form,2);
  *form = iga->form;
  PetscFunctionReturn(0);
}

PetscErrorCode IGASetForm(IGA iga,IGAForm form)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(form,2);
  if (form == iga->form) PetscFunctionReturn(0);
  ierr = IGAFormReference(form);CHKERRQ(ierr);
  ierr = IGAFormDestroy(&iga->form);CHKERRQ(ierr);
  iga->form = form;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormCreate(IGAForm *_form)
{
  PetscInt       a,s;
  IGAForm        form;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_form,1);
  ierr = PetscCalloc1(1,&form);CHKERRQ(ierr);
  *_form = form; form->refct = 1;
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

PetscErrorCode IGAFormDestroy(IGAForm *_form)
{
  PetscInt       a,s;
  IGAForm        form;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_form,1);
  form = *_form; *_form = NULL;
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

PetscErrorCode IGAFormReset(IGAForm form)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!form) PetscFunctionReturn(0);
  PetscValidPointer(form,1);
  form->dof = -1;
  ierr = PetscMemzero(form->ops,sizeof(struct _IGAFormOps));CHKERRQ(ierr);
  ierr = PetscMemzero(form->value,3*2*sizeof(struct _IGAFormBC));CHKERRQ(ierr);
  ierr = PetscMemzero(form->load,3*2*sizeof(struct _IGAFormBC));CHKERRQ(ierr);
  ierr = PetscMemzero(form->visit,3*2*sizeof(PetscBool));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  } while (0)

#define IGAFormCheckArg(arg,m) \
do { \
  if (arg<0)  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,#arg" must be nonnegative, got %d",(int)arg); \
  if (arg>=m) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,#arg" must be less than %d, got %d",(int)m,(int)arg); \
 } while (0)

static inline
void IGAFormBCSetEntry(IGAFormBC bc,PetscInt field,PetscScalar value)
{
  PetscInt k;
  for (k=0; k<bc->count; k++)
    if (bc->field[k] == field) break;
  if (k == bc->count) bc->count++;
  bc->field[k] = field;
  bc->value[k] = value;
}

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

PetscErrorCode IGAFormSetBoundaryForm(IGAForm form,PetscInt axis,PetscInt side,PetscBool flag)
{
  PetscFunctionBegin;
  IGAFormCheckArg(axis,3);
  IGAFormCheckArg(side,2);
  form->visit[axis][side] = flag ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

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

PetscErrorCode IGAFormSetVector(IGAForm form,IGAFormVector Vector,void *VecCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->Vector = Vector;
  form->ops->VecCtx = VecCtx;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormSetMatrix(IGAForm form,IGAFormMatrix Matrix,void *MatCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->Matrix = Matrix;
  form->ops->MatCtx = MatCtx;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormSetSystem(IGAForm form,IGAFormSystem System,void *SysCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->System = System;
  form->ops->SysCtx = SysCtx;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormSetFunction(IGAForm form,IGAFormFunction Function,void *FunCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->Function = Function;
  form->ops->FunCtx   = FunCtx;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormSetJacobian(IGAForm form,IGAFormJacobian Jacobian,void *JacCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->Jacobian = Jacobian;
  form->ops->JacCtx   = JacCtx;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormSetIFunction(IGAForm form,IGAFormIFunction IFunction,void *IFunCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->IFunction = IFunction;
  form->ops->IFunCtx   = IFunCtx;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormSetIJacobian(IGAForm form,IGAFormIJacobian IJacobian,void *IJacCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->IJacobian = IJacobian;
  form->ops->IJacCtx   = IJacCtx;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormSetI2Function(IGAForm form,IGAFormI2Function IFunction,void *IFunCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->I2Function = IFunction;
  form->ops->IFunCtx    = IFunCtx;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormSetI2Jacobian(IGAForm form,IGAFormI2Jacobian IJacobian,void *IJacCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->I2Jacobian = IJacobian;
  form->ops->IJacCtx    = IJacCtx;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormSetIEFunction(IGAForm form,IGAFormIEFunction IEFunction,void *IEFunCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->IEFunction = IEFunction;
  form->ops->IEFunCtx   = IEFunCtx;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormSetIEJacobian(IGAForm form,IGAFormIEJacobian IEJacobian,void *IEJacCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->IEJacobian = IEJacobian;
  form->ops->IEJacCtx   = IEJacCtx;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormSetRHSFunction(IGAForm form,IGAFormRHSFunction RHSFunction,void *RHSFunCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->RHSFunction = RHSFunction;
  form->ops->RHSFunCtx   = RHSFunCtx;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormSetRHSJacobian(IGAForm form,IGAFormRHSJacobian RHSJacobian,void *RHSJacCtx)
{
  PetscFunctionBegin;
  PetscValidPointer(form,1);
  form->ops->RHSJacobian = RHSJacobian;
  form->ops->RHSJacCtx   = RHSJacCtx;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------- */

PetscErrorCode IGASetFixTable(IGA iga,Vec U)
{
  Vec               local;
  PetscInt          nlocal;
  const PetscScalar *vlocal;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (U) PetscValidHeaderSpecific(U,VEC_CLASSID,2);
  IGACheckSetUpStage2(iga,1);

  iga->fixtable = PETSC_FALSE;
  ierr = PetscFree(iga->fixtableU);CHKERRQ(ierr);
  if (!U) PetscFunctionReturn(0);

  ierr = IGAGetLocalVecArray(iga,U,&local,&vlocal);CHKERRQ(ierr);
  ierr = VecGetSize(local,&nlocal);CHKERRQ(ierr);
  iga->fixtable = PETSC_TRUE;
  ierr = PetscMalloc1((size_t)nlocal,&iga->fixtableU);CHKERRQ(ierr);
  ierr = PetscMemcpy(iga->fixtableU,vlocal,(size_t)nlocal*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,U,&local,&vlocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_DEBUG)
#if defined(PETSC_USE_COMPLEX)
#undef  PetscValidLogicalCollectiveScalar
#define PetscValidLogicalCollectiveScalar(a,b,c)                        \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscReal b1[4],b2[4];                                              \
    b1[0] = -PetscRealPart(b);      b1[1] = PetscRealPart(b);           \
    b1[2] = -PetscImaginaryPart(b); b1[3] = PetscImaginaryPart(b);      \
    _7_ierr = MPI_Allreduce(b1,b2,4,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)a));CHKERRQ(_7_ierr); \
    if (PetscAbsReal(b2[0]+b2[1]) > 0) SETERRQ(PetscObjectComm((PetscObject)a),PETSC_ERR_ARG_WRONG,"Scalar value must be same on all processes, argument # %d",c); \
    if (PetscAbsReal(b2[2]+b2[3]) > 0) SETERRQ(PetscObjectComm((PetscObject)a),PETSC_ERR_ARG_WRONG,"Scalar value must be same on all processes, argument # %d",c); \
  } while (0)
#else
#undef  PetscValidLogicalCollectiveScalar
#define PetscValidLogicalCollectiveScalar(a,b,c)                        \
  do {                                                                  \
    PetscErrorCode _7_ierr;                                             \
    PetscReal b1[2],b2[2];                                              \
    b1[0] = -PetscRealPart(b); b1[1] = PetscRealPart(b);                \
    _7_ierr = MPI_Allreduce(b1,b2,2,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)a));CHKERRQ(_7_ierr); \
    if (PetscAbsReal(b2[0]+b2[1]) > 0) SETERRQ(PetscObjectComm((PetscObject)a),PETSC_ERR_ARG_WRONG,"Scalar value must be same on all processes, argument # %d",c); \
  } while (0)
#endif
#endif

PetscErrorCode IGASetBoundaryValue(IGA iga,PetscInt axis,PetscInt side,PetscInt field,PetscScalar value)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,axis,2);
  PetscValidLogicalCollectiveInt(iga,side,3);
  PetscValidLogicalCollectiveInt(iga,field,4);
  PetscValidLogicalCollectiveScalar(iga,value,5);
  if (iga->dim > 0) IGAFormCheckArg(axis,iga->dim);
  if (iga->dof > 0) IGAFormCheckArg(field,iga->dof);
  ierr = IGAFormSetBoundaryValue(iga->form,axis,side,field,value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGASetBoundaryLoad(IGA iga,PetscInt axis,PetscInt side,PetscInt field,PetscScalar value)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,axis,2);
  PetscValidLogicalCollectiveInt(iga,side,3);
  PetscValidLogicalCollectiveInt(iga,field,4);
  PetscValidLogicalCollectiveScalar(iga,value,5);
  if (iga->dim > 0) IGAFormCheckArg(axis,iga->dim);
  if (iga->dof > 0) IGAFormCheckArg(field,iga->dof);
  ierr = IGAFormSetBoundaryLoad(iga->form,axis,side,field,value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGASetBoundaryForm(IGA iga,PetscInt axis,PetscInt side,PetscBool flag)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,axis,2);
  PetscValidLogicalCollectiveInt(iga,side,3);
  PetscValidLogicalCollectiveBool(iga,flag,4);
  if (iga->dim > 0) IGAFormCheckArg(axis,iga->dim);
  ierr = IGAFormSetBoundaryForm(iga->form,axis,side,flag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   IGASetFormSystem - Set the user callback to form the vector
   which represents the discretized L(w).

   Logically collective on IGA

   Input Parameters:
+  iga - the IGA context
.  Vector - the function which evaluates L(w)
-  VecCtx - user-defined context for evaluation routine (may be NULL)

   Details of Vector:
$  PetscErrorCode Vector(IGAPoint p,PetscScalar F[],void *ctx);
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

/*@
   IGASetFormMatrix - Set the user callback to form the matrix
   which represents the discretized a(w,u).

   Logically collective on IGA

   Input Parameters:
+  iga - the IGA context
.  Matrix - the function which evaluates a(w,u)
-  MatCtx - user-defined context for evaluation routine (may be NULL)

   Details of System:
$  PetscErrorCode System(IGAPoint p,PetscScalar K[],void *ctx);
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

/*@
   IGASetFormSystem - Set the user callback to form the matrix and vector
   which represents the discretized a(w,u) = L(w).

   Logically collective on IGA

   Input Parameters:
+  iga - the IGA context
.  System - the function which evaluates a(w,u) and L(w)
-  SysCtx - user-defined context for evaluation routine (may be NULL)

   Details of System:
$  PetscErrorCode System(IGAPoint p,PetscScalar K[],PetscScalar F[],void *ctx);
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

/*@
   IGASetFormFunction - Set the function which computes the residual vector
   F(U)=0 for use with nonlinear problems.

   Logically Collective on IGA

   Input Parameter:
+  iga - the IGA context
.  Function - the function evaluation routine
-  FunCtx - user-defined context for private data for the function evaluation routine (may be NULL)

   Details of Function:
$  PetscErrorCode Function(IGAPoint p,const PetscScalar U[],PetscScalar F[],void *ctx);
+  p - point at which to compute the residual
.  U - local state vector
.  F - local contribution to global residual vector
-  ctx - user-defined context for evaluation routine

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

/*@
   IGASetFormIJacobian - Set the function to compute the Jacobian matrix
   J = dF/dU where F(U) is the residual function you provided with IGASetFormIFunction().

   Logically Collective on IGA

   Input Parameter:
+  iga - the IGA context
.  Jacobian - the Jacobian evaluation routine
-  JacCtx - user-defined context for private data for the Jacobian evaluation routine (may be NULL)

   Details of Jacobian:
$  PetscErrorCode Jacobian(IGAPoint p,const PetscScalar U[],PetscScalar J[],void *ctx);
+  p - point at which to compute the Jacobian
.  U - local state vector
.  J - local contribution to global Jacobian matrix
-  ctx - user-defined context for evaluation routine

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
                            PetscReal a,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar F[],void *ctx);
+  p - point at which to compute the residual
.  dt - time step size
.  a - positive parameter which depends on the time integration method (XXX Should this be here?)
.  V - time derivative of the state vector
.  t - time at step/stage being solved
.  U - state vector
.  F - function vector
-  ctx - user-defined context for evaluation routine

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
                            PetscReal a,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar J[],void *ctx);
+  p - point at which to compute the Jacobian
.  dt - time step size
.  a - positive parameter which depends on the time integration method
.  V - time derivative of the state vector
.  t - time at step/stage being solved
.  U - state vector
.  J - Jacobian matrix
-  ctx - user-defined context for evaluation routine

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
                            PetscReal a,const PetscScalar A[],
                            PetscReal v,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar F[],void *ctx);
+  p - point at which to compute the residual
.  dt - time step size
.  a - positive parameter which depends on the time integration method
.  A - second time derivative of the state vector
.  v - positive parameter which depends on the time integration method
.  V - time derivative of the state vector
.  t - time at step/stage being solved
.  U - state vector
.  F - function vector
-  ctx - user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetFormI2Function(IGA iga,IGAFormI2Function IFunction,void *IFunCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->I2Function = IFunction;
  iga->form->ops->IFunCtx    = IFunCtx;
  PetscFunctionReturn(0);
}

/*@
   IGASetFormI2Jacobian - Set the function to compute the Jacobian matrix
   J = a*dF/dU_tt + v*dF/dU_t + dF/dU  where F(t,U_tt,U_t,U) is
   the function you provided with IGASetFormI2Function().

   Logically Collective on IGA

   Input Parameter:
+  iga       - the IGA context
.  IJacobian - the Jacobian evaluation routine
-  IJacCtx   - user-defined context for private data for the Jacobian evaluation routine (may be NULL)

   Details of IJacobian:
$  PetscErrorCode IJacobian(IGAPoint p,PetscReal dt,
                            PetscReal a,const PetscScalar A[],
                            PetscReal v,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar J[],void *ctx);
+  p - point at which to compute the Jacobian
.  dt  - time step size
.  a - positive parameter which depends on the time integration method
.  A - second time derivative of the state vector
.  v - positive parameter which depends on the time integration method
.  V - time derivative of the state vector
.  t - time at step/stage being solved
.  U - state vector
.  J - Jacobian matrix
-  ctx - user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetFormI2Jacobian(IGA iga,IGAFormI2Jacobian IJacobian,void *IJacCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->I2Jacobian = IJacobian;
  iga->form->ops->IJacCtx    = IJacCtx;
  PetscFunctionReturn(0);
}

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
                             PetscReal a, const PetscScalar V[],
                             PetscReal t, const PetscScalar U[],
                             PetscReal t0,const PetscScalar U0[],
                             PetscScalar F[],void *ctx);
+  p - point at which to compute the residual
.  dt - time step size
.  a - positive parameter which depends on the time integration method (XXX Should this be here?)
.  V - time derivative of the state vector at t0
.  t - time at step/stage being solved
.  U - state vector at t
.  t0 - time at current step
.  U0 - state vector at t0
.  F - function vector
-  ctx - user-defined context for evaluation routine

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
                             PetscReal a, const PetscScalar V[],
                             PetscReal t, const PetscScalar U[],
                             PetscReal t0,const PetscScalar U0[],
                             PetscScalar J[],void *ctx);
+  p - point at which to compute the Jacobian
.  dt - time step size
.  a - positive parameter which depends on the time integration method
.  V - time derivative of the state vector at t0
.  t - time at step/stage being solved
.  U - state vector at t
.  t0 - time at current step
.  U0 - state vector at t0
.  J - Jacobian matrix
-  ctx - user-defined context for evaluation routine

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

/*@

   IGASetFormRHSFunction - Set the function to compute the right-hand side function
   F(t,U) for use with explicit or implicit time stepping routines.

   Logically Collective on IGA

   Input Parameter:
+  iga - the IGA context
.  RHSFunction - the function evaluation routine
-  RHSFunCtx - user-defined context for private data for the function evaluation routine (may be NULL)

   Details of RHSFunction:
$  PetscErrorCode RHSFunction(IGAPoint p,PetscReal dt,
                              PetscReal t, const PetscScalar U[],
                              PetscScalar F[],void *ctx);
+  p - point at which to compute the residual
.  dt - time step size
.  t - time at step/stage being solved
.  U - state vector at t
.  F - function vector
-  ctx - user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetFormRHSFunction(IGA iga,IGAFormRHSFunction RHSFunction,void *RHSFunCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->RHSFunction = RHSFunction;
  iga->form->ops->RHSFunCtx   = RHSFunCtx;
  PetscFunctionReturn(0);
}

/*@
   IGASetFormRHSJacobian - Set the function to compute the Jacobian matrix
   J = dF/dU where F(t,U) is the right-hand side function you provided with
   IGASetFormRHSFunction(). For use with implicit or explicit TS methods.

   Logically Collective on IGA

   Input Parameter:
+  iga - the IGA context
.  RHSJacobian - the Jacobian evaluation routine
-  RHSJacCtx - user-defined context for private data for the Jacobian evaluation routine (may be NULL)

   Details of RHSJacobian:
$  PetscErrorCode RHSJacobian(IGAPoint p,PetscReal dt,
                              PetscReal t, const PetscScalar U[],
                              PetscScalar J[],void *ctx);
+  p - point at which to compute the Jacobian
.  dt - time step size
.  t - time at step/stage being solved
.  U - state vector at t
.  J - Jacobian matrix
-  ctx - user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetFormRHSJacobian(IGA iga,IGAFormRHSJacobian RHSJacobian,void *RHSJacCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->form->ops->RHSJacobian = RHSJacobian;
  iga->form->ops->RHSJacCtx   = RHSJacCtx;
  PetscFunctionReturn(0);
}
