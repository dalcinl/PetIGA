#include <petscts.h>
#if PETSC_VERSION_LT(3,7,0)

#include <petscts2.h>
#if PETSC_VERSION_LT(3,6,0)
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/
#else
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#endif

#undef  __FUNCT__
#define __FUNCT__ PETSC_FUNCTION_NAME

static PetscErrorCode DMTSDuplicate(DMTS oldtsdm,DMTS newtsdm)
{
  PetscObject    oldobj = (PetscObject)oldtsdm;
  PetscObject    newobj = (PetscObject)newtsdm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFunctionListDuplicate(oldobj->qlist,&newobj->qlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSSetI2Function(DM dm,TSI2Function fun,void *ctx)
{
  DMTS           tsdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&tsdm);CHKERRQ(ierr);
  tsdm->ops->duplicate = DMTSDuplicate;
  if (fun) {ierr = PetscObjectComposeFunction((PetscObject)tsdm,"TSI2Function_C",fun);CHKERRQ(ierr);}
  if (ctx) {ierr = DMTSSetIFunction(dm,NULL,ctx);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSSetI2Jacobian(DM dm,TSI2Jacobian jac,void *ctx)
{
  DMTS           tsdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&tsdm);CHKERRQ(ierr);
  tsdm->ops->duplicate = DMTSDuplicate;
  if (jac) {ierr = PetscObjectComposeFunction((PetscObject)tsdm,"TSI2Jacobian_C",jac);CHKERRQ(ierr);}
  if (ctx) {ierr = DMTSSetIJacobian(dm,NULL,ctx);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSGetI2Function(DM dm,TSI2Function *fun,void **ctx)
{
  DMTS           tsdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm,&tsdm);CHKERRQ(ierr);
  if (fun) {ierr = PetscObjectQueryFunction((PetscObject)tsdm,"TSI2Function_C",(PetscVoidFunction*)fun);CHKERRQ(ierr);}
  if (ctx) {ierr = DMTSGetIFunction(dm,NULL,ctx);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSGetI2Jacobian(DM dm,TSI2Jacobian *jac,void **ctx)
{
  DMTS           tsdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm,&tsdm);CHKERRQ(ierr);
  if (jac) {ierr = PetscObjectQueryFunction((PetscObject)tsdm,"TSI2Jacobian_C",(PetscVoidFunction*)jac);CHKERRQ(ierr);}
  if (ctx) {ierr = DMTSGetIJacobian(dm,NULL,ctx);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

/*@C
   TSSetI2Function - Set the function to compute F(t,U,U_t,U_tt) where F = 0 is the DAE to be solved.

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  F   - vector to hold the residual (or NULL to have it created internally)
.  fun - the function evaluation routine
-  ctx - user-defined context for private data for the function evaluation routine (may be NULL)

   Calling sequence of fun:
$  fun(TS ts,PetscReal t,Vec U,Vec U_t,Vec U_tt,Vec F,ctx);

+  t    - time at step/stage being solved
.  U    - state vector
.  U_t  - time derivative of state vector
.  U_tt - second time derivative of state vector
.  F    - function vector
-  ctx  - [optional] user-defined context for matrix evaluation routine (may be NULL)

   Level: beginner

.keywords: TS, TSALPHA2, timestep, set, ODE, DAE, Function

.seealso: TSALPHA2, TSSetI2Jacobian()
@*/
PetscErrorCode TSSetI2Function(TS ts,Vec F,TSI2Function fun,void *ctx)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (F) PetscValidHeaderSpecific(F,VEC_CLASSID,2);
  if (F) {ierr = TSSetIFunction(ts,F,NULL,NULL);CHKERRQ(ierr);}
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetI2Function(dm,fun,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSSetIJacobian - Set the function to compute the matrix dF/dU + v*dF/dU_t  + a*dF/dU_tt
        where F(t,U,U_t,U_tt) is the function you provided with TSSetI2Function().

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  J   - Jacobian matrix
.  P   - preconditioning matrix for J (may be same as J)
.  jac - the Jacobian evaluation routine
-  ctx - user-defined context for private data for the Jacobian evaluation routine (may be NULL)

   Calling sequence of jac:
$  jac(TS ts,PetscReal t,Vec U,Vec U_t,Vec U_tt,PetscReal v,PetscReal a,Mat *J,Mat *P,MatStructure *m,void *ctx);

+  t    - time at step/stage being solved
.  U    - state vector
.  U_t  - time derivative of state vector
.  U_tt - second time derivative of state vector
.  v    - shift for U_t
.  a    - shift for U_tt
.  J    - Jacobian of G(U) = F(t,U,W+v*U,W'+a*U), equivalent to dF/dU + v*dF/dU_t  + a*dF/dU_tt
.  P    - preconditioning matrix for J, may be same as J
.  m    - flag indicating information about the preconditioner matrix
          structure (same as flag in KSPSetOperators())
-  ctx  - [optional] user-defined context for matrix evaluation routine

   Notes:
   The matrices J and P are exactly the matrices that are used by SNES for the nonlinear solve.

   The matrix dF/dU + v*dF/dU_t + a*dF/dU_tt you provide turns out to be
   the Jacobian of G(U) = F(t,U,W+v*U,W'+a*U) where F(t,U,U_t,U_tt) = 0 is the DAE to be solved.
   The time integrator internally approximates U_t by W+v*U and U_tt by W'+a*U  where the positive "shift"
   parameters 'a' and 'b' and vectors W, W' depend on the integration method, step size, and past states.

   Level: beginner

.keywords: TS, TSALPHA2, timestep, set, ODE, DAE, Jacobian

.seealso: TSALPHA2, TSSetI2Function()
@*/
PetscErrorCode TSSetI2Jacobian(TS ts,Mat J,Mat P,TSI2Jacobian jac,void *ctx)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (J) PetscValidHeaderSpecific(J,MAT_CLASSID,2);
  if (P) PetscValidHeaderSpecific(P,MAT_CLASSID,3);
  if (J || P) {ierr = TSSetIJacobian(ts,J,P,NULL,NULL);CHKERRQ(ierr);}
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetI2Jacobian(dm,jac,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSComputeI2Function(TS ts,PetscReal t,Vec X,Vec V,Vec A,Vec F)
{
  DM             dm;
  TSI2Function   I2Function;
  void           *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(A,VEC_CLASSID,5);
  PetscValidHeaderSpecific(F,VEC_CLASSID,6);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetI2Function(dm,&I2Function,&ctx);CHKERRQ(ierr);
  if (I2Function) {
    PetscStackPush("TS user implicit function");
    ierr = I2Function(ts,t,X,V,A,F,ctx);CHKERRQ(ierr);
    PetscStackPop;
  } else {
    ierr = TSComputeIFunction(ts,t,X,A,F,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TSComputeI2Jacobian(TS ts,PetscReal t,Vec X,Vec V,Vec A,PetscReal shiftV,PetscReal shiftA,Mat J,Mat P)
{
  DM             dm;
  TSI2Jacobian   I2Jacobian;
  void           *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(A,VEC_CLASSID,5);
  PetscValidHeaderSpecific(J,MAT_CLASSID,8);
  PetscValidHeaderSpecific(P,MAT_CLASSID,9);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetI2Jacobian(dm,&I2Jacobian,&ctx);CHKERRQ(ierr);
  if (I2Jacobian) {
    PetscStackPush("TS user implicit Jacobian");
    ierr = I2Jacobian(ts,t,X,V,A,shiftV,shiftA,J,P,ctx);CHKERRQ(ierr);
    PetscStackPop;
  } else {
#if PETSC_VERSION_LT(3,5,0)
    MatStructure m;
    ierr = TSComputeIJacobian(ts,t,X,A,shiftA,&J,&P,&m,PETSC_FALSE);CHKERRQ(ierr);
#else
    ierr = TSComputeIJacobian(ts,t,X,A,shiftA,J,P,PETSC_FALSE);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

/*@
   TS2SetSolution - Sets the initial solution and time-derivative vectors
   for use by the TSALPHA2 routines.

   Logically Collective on TS and Vec

   Input Parameters:
+  ts - the TS context
.  X - the solution vector
-  V - the time-derivative vector

   Level: beginner

.keywords: TS, TSALPHA2, timestep, set, solution, initial conditions
@*/
PetscErrorCode TS2SetSolution(TS ts,Vec X,Vec V)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(V,VEC_CLASSID,3);
  ierr = PetscUseMethod(ts,"TS2SetSolution_C",(TS,Vec,Vec),(ts,X,V));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TS2GetSolution - Returns the solution and time-derivative vectors
   at the present timestep. It is valid to call this routine inside
   the function that you are evaluating in order to move to the new
   timestep. This vector not changed until the solution at the next
   timestep has been calculated.

   Not Collective, but Vec returned is parallel if TS is parallel

   Input Parameter:
.  ts - the TS context

   Output Parameter:
+  X - the vector containing the solution
-  V - the vector containing the time-derivative

   Level: intermediate

.seealso: TSGetTimeStep()

.keywords: TS, TSALPHA2, timestep, get, solution
@*/
PetscErrorCode TS2GetSolution(TS ts,Vec *X, Vec *V)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (X) PetscValidPointer(X,2);
  if (V) PetscValidPointer(V,3);
  ierr = PetscUseMethod(ts,"TS2GetSolution_C",(TS,Vec*,Vec*),(ts,X,V));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif/* PETSc >= 3.7 */
