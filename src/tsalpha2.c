#include <petscts2.h>
#if PETSC_VERSION_(3,2,0)
#include <private/tsimpl.h>                      /*I   "petscts.h"   I*/
#else
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/
#endif

#if PETSC_VERSION_(3,2,0)
#define PetscObjectTypeCompare PetscTypeCompare
#define TSPreStage(ts,t) (0)
#define ksp_its  linear_its
#define snes_its nonlinear_its
#endif

typedef struct {

  PetscReal stage_time;
  PetscReal scale_F;
  PetscReal shift_V;
  PetscReal shift_A;
  Vec       vec_sol_X;
  Vec       vec_sol_V;
  Vec       X0,Xa,X1;
  Vec       V0,Va,V1;
  Vec       A0,Aa,A1;
  int      (*StageTime)(TS);
  int      (*StageVecs)(TS,Vec);

  TSIFunction2 Function;
  void         *FunCtx;
  TSIJacobian2 Jacobian;
  void         *JacCtx;

  PetscReal Alpha_m;
  PetscReal Alpha_f;
  PetscReal Beta;
  PetscReal Gamma;

} TS_Alpha2;

PETSC_EXTERN PetscLogEvent TS_FunctionEval;
PETSC_EXTERN PetscLogEvent TS_JacobianEval;

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormFunction_Alpha2"
static PetscErrorCode SNESTSFormFunction_Alpha2(SNES snes,Vec X,Vec F,TS ts)
{
  TS_Alpha2      *th = (TS_Alpha2*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = th->StageVecs(ts,X);CHKERRQ(ierr);
  if (th->Function) {
    /* F = Function(ta,Xa,Va,Aa) */
    PetscReal ta = th->stage_time;
    Vec       Xa = th->Xa, Va = th->Va, Aa = th->Aa;
    ierr = TSComputeIFunction2(ts,ta,Xa,Va,Aa,F,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    /* F = Function(ta,Xa,Aa) */
    PetscReal ta = th->stage_time;
    Vec       Xa = th->Xa, Aa = th->Aa;
    ierr = TSComputeIFunction(ts,ta,Xa,Aa,F,PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = VecScale(F,th->scale_F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormJacobian_Alpha2"
static PetscErrorCode SNESTSFormJacobian_Alpha2(SNES snes,Vec X,Mat *J,Mat *P,MatStructure *str,TS ts)
{
  TS_Alpha2      *th = (TS_Alpha2*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = th->StageVecs(ts,X);CHKERRQ(ierr);
  if (th->Jacobian) {
    /* J,P = Jacobian(ta,Xa,Va,Aa) */
    PetscReal ta = th->stage_time;
    Vec       Xa = th->Xa, Va = th->Va, Aa = th->Aa;
    PetscReal dVdX = th->shift_V, dAdX = th->shift_A;
    ierr = TSComputeIJacobian2(ts,ta,Xa,Va,Aa,dVdX,dAdX,J,P,str,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    /* J,P = Jacobian(ta,Xa,Aa) */
    PetscReal ta = th->stage_time;
    Vec       Xa = th->Xa, Aa = th->Aa;
    PetscReal dAdX = th->shift_A;
    ierr = TSComputeIJacobian(ts,ta,Xa,Aa,dAdX,J,P,str,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(*J,MAT_CLASSID,3);
  PetscValidHeaderSpecific(*P,MAT_CLASSID,4);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSStep_Alpha2"
static PetscErrorCode TSStep_Alpha2(TS ts)
{
  TS_Alpha2           *th    = (TS_Alpha2*)ts->data;
  PetscInt            its,lits,reject;
  PetscReal           next_time_step;
  SNESConvergedReason snesreason = SNES_CONVERGED_ITERATING;
  PetscErrorCode      ierr;

  PetscFunctionBegin;

  if (ts->steps == 0) {
    ierr = VecSet(th->A0,0.0);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(th->A1,th->A0);CHKERRQ(ierr);
  }
  ierr = VecCopy(th->vec_sol_X,th->X0);CHKERRQ(ierr);
  ierr = VecCopy(th->vec_sol_V,th->V0);CHKERRQ(ierr);
  ierr = TSPreStep(ts);CHKERRQ(ierr);

  next_time_step = ts->time_step;
  for (reject=0; reject<ts->max_reject; reject++,ts->reject++) {
    ts->time_step  = next_time_step;
    ierr = th->StageTime(ts);CHKERRQ(ierr);
    ierr = TSPreStage(ts,th->stage_time);CHKERRQ(ierr);
    /* nonlinear solve R(X,V,A) = 0 */
    ierr = VecCopy(th->X0,th->X1);CHKERRQ(ierr);
    ierr = SNESSolve(ts->snes,PETSC_NULL,th->X1);CHKERRQ(ierr);
    ierr = th->StageVecs(ts,th->X1);CHKERRQ(ierr);
    /* nonlinear solve convergence */
    ierr = SNESGetConvergedReason(ts->snes,&snesreason);CHKERRQ(ierr);
    if (snesreason < 0 /*&& !th->adapt*/) break;
    ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
    ts->snes_its += its; ts->ksp_its += lits;
    ts->snes_its += its; ts->ksp_its += lits;
    ierr = PetscInfo3(ts,"step=%D, nonlinear solve iterations=%D, linear solve iterations=%D\n",ts->steps,its,lits);CHKERRQ(ierr);
    /* time step adaptativity */
    break; /* XXX Implement time adaptativity !!! */
  }
  if (snesreason < 0 && ts->max_snes_failures > 0 && ++ts->num_snes_failures >= ts->max_snes_failures) {
    ts->reason = TS_DIVERGED_NONLINEAR_SOLVE;
    ierr = PetscInfo2(ts,"Step=%D, nonlinear solve solve failures %D greater than current TS allowed, stopping solve\n",ts->steps,ts->num_snes_failures);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (reject >= ts->max_reject) {
    ts->reason = TS_DIVERGED_STEP_REJECTED;
    ierr = PetscInfo2(ts,"Step=%D, step rejections %D greater than current TS allowed, stopping solve\n",ts->steps,reject);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = VecCopy(th->X1,th->vec_sol_X);CHKERRQ(ierr);
  ierr = VecCopy(th->V1,th->vec_sol_V);CHKERRQ(ierr);
  ts->ptime += ts->time_step;
  ts->time_step = next_time_step;
  ts->steps++;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSUpdateStageTime_Alpha2"
static PetscErrorCode TSUpdateStageTime_Alpha2(TS ts)
{
  TS_Alpha2 *th = (TS_Alpha2*)ts->data;
  PetscReal t = ts->ptime;
  PetscReal dt = ts->time_step;
  PetscReal Gamma   = th->Gamma;
  PetscReal Beta    = th->Beta;
  PetscReal Alpha_m = th->Alpha_m;
  PetscReal Alpha_f = th->Alpha_f;
  PetscFunctionBegin;
  th->stage_time = t + Alpha_f*dt;
  th->scale_F = 1/Alpha_f;
  th->shift_V = Gamma/(dt*Beta);
  th->shift_A = Alpha_m/(Alpha_f*dt*dt*Beta);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSUpdateStageVecs_Alpha2"
static PetscErrorCode TSUpdateStageVecs_Alpha2(TS ts,Vec X)
{
  TS_Alpha2      *th = (TS_Alpha2*)ts->data;
  Vec            X1 = X,      V1 = th->V1, A1 = th->A1;
  Vec            Xa = th->Xa, Va = th->Va, Aa = th->Aa;
  Vec            X0 = th->X0, V0 = th->V0, A0 = th->A0;
  PetscReal      dt = ts->time_step;
  PetscReal      Gamma   = th->Gamma;
  PetscReal      Beta    = th->Beta;
  PetscReal      Alpha_m = th->Alpha_m;
  PetscReal      Alpha_f = th->Alpha_f;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* A1 = ... */
  ierr = VecWAXPY(A1,-1.0,X0,X1);CHKERRQ(ierr);
  ierr = VecAXPY (A1,-dt,V0);CHKERRQ(ierr);
  ierr = VecAXPBY(A1,-(1-2*Beta)/(2*Beta),1/(dt*dt*Beta),A0);CHKERRQ(ierr);
  /* V1 = ... */
  ierr = VecWAXPY(V1,(1.0-Gamma)/Gamma,A0,A1);CHKERRQ(ierr);
  ierr = VecAYPX (V1,dt*Gamma,V0);CHKERRQ(ierr);

  /* Xa = X0 + Alpha_f*(X1-X0) */
  ierr = VecWAXPY(Xa,-1.0,X0,X1);CHKERRQ(ierr);
  ierr = VecAYPX (Xa,Alpha_f,X0);CHKERRQ(ierr);
  /* Va = V0 + Alpha_f*(V1-V0) */
  ierr = VecWAXPY(Va,-1.0,V0,V1);CHKERRQ(ierr);
  ierr = VecAYPX (Va,Alpha_f,V0);CHKERRQ(ierr);
  /* Aa = A0 + Alpha_m*(A1-A0) */
  ierr = VecWAXPY(Aa,-1.0,A0,A1);CHKERRQ(ierr);
  ierr = VecAYPX (Aa,Alpha_m,A0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSReset_Alpha2"
static PetscErrorCode TSReset_Alpha2(TS ts)
{
  TS_Alpha2      *th = (TS_Alpha2*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&th->X0);CHKERRQ(ierr);
  ierr = VecDestroy(&th->Xa);CHKERRQ(ierr);
  ierr = VecDestroy(&th->X1);CHKERRQ(ierr);
  ierr = VecDestroy(&th->V0);CHKERRQ(ierr);
  ierr = VecDestroy(&th->Va);CHKERRQ(ierr);
  ierr = VecDestroy(&th->V1);CHKERRQ(ierr);
  ierr = VecDestroy(&th->A0);CHKERRQ(ierr);
  ierr = VecDestroy(&th->Aa);CHKERRQ(ierr);
  ierr = VecDestroy(&th->A1);CHKERRQ(ierr);
  ierr = VecDestroy(&th->vec_sol_X);CHKERRQ(ierr);
  ierr = VecDestroy(&th->vec_sol_V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_Alpha2"
static PetscErrorCode TSDestroy_Alpha2(TS ts)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TSReset_Alpha2(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSetIFunction2_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSetIJacobian2_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSComputeIFunction2_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSComputeIJacobian2_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSetSolution2_C",   "",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSGetSolution2_C",   "",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSolve2_C",         "",PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSAlpha2SetRadius_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSAlpha2SetParams_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSAlpha2GetParams_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_Alpha2"
static PetscErrorCode TSSetUp_Alpha2(TS ts)
{
  TS_Alpha2      *th = (TS_Alpha2*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!th->vec_sol_X) {
    ierr = PetscObjectReference((PetscObject)ts->vec_sol);CHKERRQ(ierr);
    th->vec_sol_X = ts->vec_sol;
  }
  if (!th->vec_sol_V) {
    ierr = VecDuplicate(ts->vec_sol,&th->vec_sol_V);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(ts->vec_sol,&th->X0);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Xa);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->X1);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->V0);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Va);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->V1);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->A0);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Aa);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->A1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_Alpha2"
static PetscErrorCode TSSetFromOptions_Alpha2(TS ts)
{
  TS_Alpha2      *th = (TS_Alpha2*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Generalized-Alpha ODE solver options");CHKERRQ(ierr);
  {
    PetscBool flg;
    PetscReal radius = 1.0;
    ierr = PetscOptionsReal("-ts_alpha_radius", "spectral radius (high-frequency dissipation)","TSAlpha2SetRadius",radius,&radius,&flg);CHKERRQ(ierr);
    if (flg) {ierr = TSAlpha2SetRadius(ts,radius);CHKERRQ(ierr);}
    ierr = PetscOptionsReal("-ts_alpha_alpha_m","algoritmic parameter alpha_m","TSAlpha2SetParams",th->Alpha_m,&th->Alpha_m,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_alpha_f","algoritmic parameter alpha_f","TSAlpha2SetParams",th->Alpha_f,&th->Alpha_f,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_gamma",  "algoritmic parameter gamma",  "TSAlpha2SetParams",th->Gamma,  &th->Gamma,  PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_beta",   "algoritmic parameter beta",   "TSAlpha2SetParams",th->Beta,   &th->Beta,   PETSC_NULL);CHKERRQ(ierr);
    ierr = TSAlpha2SetParams(ts,th->Alpha_m,th->Alpha_f,th->Gamma,th->Beta);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&ts->snes);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(ts->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSView_Alpha2"
static PetscErrorCode TSView_Alpha2(TS ts,PetscViewer viewer)
{
  TS_Alpha2      *th = (TS_Alpha2*)ts->data;
  PetscBool      ascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&ascii);CHKERRQ(ierr);
  if (ascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Alpha_m=%G, Alpha_f=%G, Gamma=%G, Beta=%G\n",th->Alpha_m,th->Alpha_f,th->Gamma,th->Beta);CHKERRQ(ierr);
  }
  ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

EXTERN_C_BEGIN

#undef __FUNCT__
#define __FUNCT__ "TSSetIFunction2_Alpha2"
PetscErrorCode TSSetIFunction2_Alpha2(TS ts,Vec F,TSIFunction2 f,void *ctx)
{
  TS_Alpha2      *th = (TS_Alpha2*)ts->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TSSetIFunction(ts,F,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  if (f)   th->Function = f;
  if (ctx) th->FunCtx   = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetIJacobian2_Alpha2"
PetscErrorCode TSSetIJacobian2_Alpha2(TS ts,Mat J,Mat P,TSIJacobian2 j,void *ctx)
{
  TS_Alpha2      *th = (TS_Alpha2*)ts->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TSSetIJacobian(ts,J,P,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  if (j)   th->Jacobian = j;
  if (ctx) th->JacCtx   = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeIFunction2_Alpha2"
PetscErrorCode TSComputeIFunction2_Alpha2(TS ts,PetscReal t,Vec X,Vec V,Vec A,Vec F,PetscBool imex)
{
  TS_Alpha2      *th = (TS_Alpha2*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!th->Function) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_USER,"Must call TSSetIFunction2()");
  ierr = PetscLogEventBegin(TS_FunctionEval,ts,X,V,F);CHKERRQ(ierr);
  PetscStackPush("TS user implicit function");
  ierr = (*th->Function)(ts,t,X,V,A,F,th->FunCtx);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(TS_FunctionEval,ts,X,V,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeIJacobian2_Alpha2"
PetscErrorCode TSComputeIJacobian2_Alpha2(TS ts,PetscReal t,Vec X,Vec V,Vec A,PetscReal shiftV,PetscReal shiftA,Mat *J,Mat *P,MatStructure *str,PetscBool imex)
{
  TS_Alpha2      *th = (TS_Alpha2*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!th->Jacobian) SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_USER,"Must call TSSetIJacobian2()");
  *str = DIFFERENT_NONZERO_PATTERN;
  ierr = PetscLogEventBegin(TS_JacobianEval,ts,X,V,*J);CHKERRQ(ierr);
  PetscStackPush("TS user implicit Jacobian");
  ierr = (*th->Jacobian)(ts,t,X,V,A,shiftV,shiftA,J,P,str,th->JacCtx);CHKERRQ(ierr);
  PetscStackPop;
  ierr = PetscLogEventEnd(TS_JacobianEval,ts,X,V,*J);CHKERRQ(ierr);
  PetscValidHeaderSpecific(*J,MAT_CLASSID,8);
  PetscValidHeaderSpecific(*P,MAT_CLASSID,9);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetSolution2_Alpha2"
PetscErrorCode TSSetSolution2_Alpha2(TS ts,Vec X,Vec V)
{
  TS_Alpha2      *th = (TS_Alpha2*)ts->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  /* set X */
  ierr = PetscObjectReference((PetscObject)X);CHKERRQ(ierr);
  ierr = VecDestroy(&th->vec_sol_X);CHKERRQ(ierr);
  th->vec_sol_X = X;
  /* set V */
  ierr = PetscObjectReference((PetscObject)V);CHKERRQ(ierr);
  ierr = VecDestroy(&th->vec_sol_V);CHKERRQ(ierr);
  th->vec_sol_V = V;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetSolution2_Alpha2"
PetscErrorCode TSGetSolution2_Alpha2(TS ts,Vec *X, Vec *V)
{
  TS_Alpha2 *th = (TS_Alpha2*)ts->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (X && !th->vec_sol_X && ts->vec_sol) {
    ierr = PetscObjectReference((PetscObject)ts->vec_sol);CHKERRQ(ierr);
    th->vec_sol_X = ts->vec_sol;
  }
  if (V && !th->vec_sol_V && ts->vec_sol) {
    ierr = VecDuplicate(ts->vec_sol,&th->vec_sol_V);CHKERRQ(ierr);
  }
  if (X) *X = th->vec_sol_X;
  if (V) *V = th->vec_sol_V;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSolve2_Alpha2"
PetscErrorCode TSSolve2_Alpha2(TS ts,Vec X,Vec V)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TSSetSolution2(ts,X,V);CHKERRQ(ierr);
#if PETSC_VERSION_(3,3,0) || PETSC_VERSION_(3,2,0)
  ierr = TSSolve(ts,X,PETSC_NULL);CHKERRQ(ierr);
#else
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

EXTERN_C_END

/* ------------------------------------------------------------ */

EXTERN_C_BEGIN

#undef __FUNCT__
#define __FUNCT__ "TSAlpha2SetRadius_Alpha2"
PetscErrorCode TSAlpha2SetRadius_Alpha2(TS ts,PetscReal radius)
{
  TS_Alpha2 *th = (TS_Alpha2*)ts->data;
  PetscFunctionBegin;
  th->Alpha_m = (2-radius)/(1+radius);
  th->Alpha_f = 1/(1+radius);
  th->Gamma   = 0.5 + th->Alpha_m - th->Alpha_f;
  th->Beta    = 0.5 * (1 + th->Alpha_m - th->Alpha_f); th->Beta *= th->Beta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlpha2SetParams_Alpha2"
PetscErrorCode TSAlpha2SetParams_Alpha2(TS ts,PetscReal alpha_m,PetscReal alpha_f,PetscReal gamma,PetscReal beta)
{
  TS_Alpha2 *th = (TS_Alpha2*)ts->data;
  PetscFunctionBegin;
  th->Alpha_m = alpha_m;
  th->Alpha_f = alpha_f;
  th->Gamma   = gamma;
  th->Beta    = beta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlpha2GetParams_Alpha2"
PetscErrorCode TSAlpha2GetParams_Alpha2(TS ts,PetscReal *alpha_m,PetscReal *alpha_f,PetscReal *gamma,PetscReal *beta)
{
  TS_Alpha2 *th = (TS_Alpha2*)ts->data;
  PetscFunctionBegin;
  if (alpha_m) *alpha_m = th->Alpha_m;
  if (alpha_f) *alpha_f = th->Alpha_f;
  if (gamma)   *gamma   = th->Gamma;
  if (beta)    *beta    = th->Beta;
  PetscFunctionReturn(0);
}

EXTERN_C_END

/* ------------------------------------------------------------ */
/*MC
      TSALPHA2 - DAE solver using the implicit Generalized-Alpha method
                 for second-order systems.

  Level: beginner

  References:
  J. Chung, G.M.Hubert. "A Time Integration Algorithm for Structural
  Dynamics with Improved Numerical Dissipation: The Generalized-alpha
  Method" ASME Journal of Applied Mechanics, 60, 371:375, 1993.

.seealso:  TSCreate(), TS, TSSetType()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSCreate_Alpha2"
PetscErrorCode TSCreate_Alpha2(TS ts)
{
  TS_Alpha2      *th;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ts->ops->step           = TSStep_Alpha2;
  ts->ops->snesfunction   = SNESTSFormFunction_Alpha2;
  ts->ops->snesjacobian   = SNESTSFormJacobian_Alpha2;

  ts->ops->reset          = TSReset_Alpha2;
  ts->ops->destroy        = TSDestroy_Alpha2;
  ts->ops->setup          = TSSetUp_Alpha2;
  ts->ops->view           = TSView_Alpha2;
  ts->ops->setfromoptions = TSSetFromOptions_Alpha2;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSetIFunction2_C","TSSetIFunction2_Alpha2",TSSetIFunction2_Alpha2);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSetIJacobian2_C","TSSetIJacobian2_Alpha2",TSSetIJacobian2_Alpha2);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSComputeIFunction2_C","TSComputeIFunction2_Alpha2",TSComputeIFunction2_Alpha2);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSComputeIJacobian2_C","TSComputeIJacobian2_Alpha2",TSComputeIJacobian2_Alpha2);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSetSolution2_C","TSSetSolution2_Alpha2",TSSetSolution2_Alpha2);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSGetSolution2_C","TSGetSolution2_Alpha2",TSGetSolution2_Alpha2);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSSolve2_C","TSSolve2_Alpha2",TSSolve2_Alpha2);CHKERRQ(ierr);

  ierr = PetscNewLog(ts,TS_Alpha2,&th);CHKERRQ(ierr);
  ts->data = (void*)th;
  th->Alpha_m = 0.5;
  th->Alpha_f = 0.5;
  th->Gamma   = 0.5;
  th->Beta    = 0.25;
  th->StageTime = TSUpdateStageTime_Alpha2;
  th->StageVecs = TSUpdateStageVecs_Alpha2;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSAlpha2SetRadius_C","TSAlpha2SetRadius_Alpha2",TSAlpha2SetRadius_Alpha2);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSAlpha2SetParams_C","TSAlpha2SetParams_Alpha2",TSAlpha2SetParams_Alpha2);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSAlpha2GetParams_C","TSAlpha2GetParams_Alpha2",TSAlpha2GetParams_Alpha2);CHKERRQ(ierr);

  if (ts->exact_final_time == PETSC_DECIDE) ts->exact_final_time = PETSC_FALSE;

  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ------------------------------------------------------------ */

#undef __FUNCT__
#define __FUNCT__ "TSSetIFunction2"
/*@C
   TSSetIFunction2 - Set the function to compute F(t,U,U_t,U_tt) where F = 0 is the DAE to be solved.

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  F   - vector to hold the residual (or PETSC_NULL to have it created internally)
.  fun - the function evaluation routine
-  ctx - user-defined context for private data for the function evaluation routine (may be PETSC_NULL)

   Calling sequence of fun:
$  fun(TS ts,PetscReal t,Vec U,Vec U_t,Vec U_tt,Vec F,ctx);

+  t    - time at step/stage being solved
.  U    - state vector
.  U_t  - time derivative of state vector
.  U_tt - second time derivative of state vector
.  F    - function vector
-  ctx  - [optional] user-defined context for matrix evaluation routine (may be PETSC_NULL)

   Level: beginner

.keywords: TS, timestep, set, ODE, DAE, Jacobian

.seealso: TSSetIJacobian2()
@*/
PetscErrorCode TSSetIFunction2(TS ts,Vec F,TSIFunction2 fun,void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (F) PetscValidHeaderSpecific(F,VEC_CLASSID,2);
  ierr = PetscUseMethod(ts,"TSSetIFunction2_C",(TS,Vec,TSIFunction2,void*),(ts,F,fun,ctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetIJacobian2"
/*@C
   TSSetIJacobian - Set the function to compute the matrix dF/dU + v*dF/dU_t  + a*dF/dU_tt
        where F(t,U,U_t,U_tt) is the function you provided with TSSetIFunction2().

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  J   - Jacobian matrix
.  P   - preconditioning matrix for J (may be same as J)
.  jac - the Jacobian evaluation routine
-  ctx - user-defined context for private data for the Jacobian evaluation routine (may be PETSC_NULL)

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

.keywords: TS, timestep, DAE, Jacobian

.seealso: TSSetIFunction2()

@*/
PetscErrorCode TSSetIJacobian2(TS ts,Mat J,Mat P,TSIJacobian2 jac,void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (J) PetscValidHeaderSpecific(J,MAT_CLASSID,2);
  if (P) PetscValidHeaderSpecific(P,MAT_CLASSID,3);
  ierr = PetscUseMethod(ts,"TSSetIJacobian2_C",(TS,Mat,Mat,TSIJacobian2,void*),(ts,J,P,jac,ctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeIFunction2"
PetscErrorCode TSComputeIFunction2(TS ts,PetscReal t,Vec X,Vec V,Vec A,Vec F,PetscBool imex)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(A,VEC_CLASSID,5);
  PetscValidHeaderSpecific(F,VEC_CLASSID,6);
  ierr = PetscUseMethod(ts,"TSComputeIFunction2_C",
                        (TS,PetscReal,Vec,Vec,Vec,Vec,PetscBool),
                        (ts,t,X,V,A,F,imex));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeIJacobian2"
PetscErrorCode TSComputeIJacobian2(TS ts,PetscReal t,Vec X,Vec V,Vec A,PetscReal shiftV,PetscReal shiftA,Mat *J,Mat *P,MatStructure *str,PetscBool imex)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(A,VEC_CLASSID,5);
  PetscValidPointer(J,8);
  PetscValidHeaderSpecific(*J,MAT_CLASSID,8);
  PetscValidPointer(P,9);
  PetscValidHeaderSpecific(*P,MAT_CLASSID,9);
  PetscValidPointer(str,10);
  ierr = PetscUseMethod(ts,"TSComputeIJacobian2_C",
                        (TS,PetscReal,Vec,Vec,Vec,PetscReal,PetscReal,Mat*,Mat*,MatStructure*,PetscBool),
                        (ts,t,X,V,A,shiftV,shiftA,J,P,str,imex));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetSolution2"
PetscErrorCode TSSetSolution2(TS ts,Vec X,Vec V)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(V,VEC_CLASSID,3);
  ierr = PetscUseMethod(ts,"TSSetSolution2_C",(TS,Vec,Vec),(ts,X,V));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetSolution2"
PetscErrorCode TSGetSolution2(TS ts,Vec *X, Vec *V)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (X) PetscValidPointer(X,2);
  if (V) PetscValidPointer(V,3);
  ierr = PetscUseMethod(ts,"TSGetSolution2_C",(TS,Vec*,Vec*),(ts,X,V));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSolve2"
PetscErrorCode TSSolve2(TS ts,Vec X,Vec V)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(V,VEC_CLASSID,3);
  ierr = PetscUseMethod(ts,"TSSolve2_C",(TS,Vec,Vec),(ts,X,V));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

#undef __FUNCT__
#define __FUNCT__ "TSAlpha2SetRadius"
/*@
  TSAlpha2SetRadius - sets the desired spectral radius of the method
                      (i.e. high-frequency numerical damping)

  Logically Collective on TS

  The algorithmic parameters \alpha_m and \alpha_f of the
  generalized-\alpha method can be computed in terms of a specified
  spectral radius \rho in [0,1] for infinite time step in order to
  control high-frequency numerical damping:
    \alpha_m = (2-\rho)/(1+\rho)
    \alpha_f = 1/(1+\rho)

  Input Parameter:
+  ts - timestepping context
-  radius - the desired spectral radius

  Options Database:
.  -ts_alpha_radius <radius>

  Level: intermediate

.seealso: TSAlpha2SetParams(), TSAlpha2GetParams()
@*/
PetscErrorCode TSAlpha2SetRadius(TS ts,PetscReal radius)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,radius,2);
  if (radius < 0 || radius > 1) SETERRQ1(((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Radius %G not in range [0,1]",radius);
  ierr = PetscTryMethod(ts,"TSAlpha2SetRadius_C",(TS,PetscReal),(ts,radius));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlpha2SetParams"
/*@
  TSAlpha2SetParams - sets the algorithmic parameters for TSALPHA2

  Not Collective

  Second-order accuracy can be obtained so long as:
    \gamma = 1/2 + alpha_m - alpha_f
    \beta  = 1/4 (1 + alpha_m - alpha_f)^2

  Unconditional stability requires:
    \alpha_m >= \alpha_f >= 1/2


  Input Parameter:
+  ts - timestepping context
.  \alpha_m - algorithmic paramenter
.  \alpha_f - algorithmic paramenter
.  \gamma   - algorithmic paramenter
-  \beta    - algorithmic paramenter

   Options Database:
+  -ts_alpha_alpha_m <alpha_m>
.  -ts_alpha_alpha_f <alpha_f>
.  -ts_alpha_gamma   <gamma>
-  -ts_alpha_beta    <beta>

  Note:
  Use of this function is normally only required to hack TSGALPHA to
  use a modified integration scheme. Users should call
  TSAlpha2SetRadius() to set the desired spectral radius of the methods
  (i.e. high-frequency damping) in order so select optimal values for
  these parameters.

  Level: advanced

.seealso: TSAlpha2SetRadius(), TSAlpha2GetParams()
@*/
PetscErrorCode TSAlpha2SetParams(TS ts,PetscReal alpha_m,PetscReal alpha_f,PetscReal gamma,PetscReal beta)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,alpha_m,2);
  PetscValidLogicalCollectiveReal(ts,alpha_f,3);
  PetscValidLogicalCollectiveReal(ts,gamma,  4);
  PetscValidLogicalCollectiveReal(ts,beta,   5);
  ierr = PetscTryMethod(ts,"TSAlpha2SetParams_C",(TS,PetscReal,PetscReal,PetscReal,PetscReal),(ts,alpha_m,alpha_f,gamma,beta));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlpha2GetParams"
/*@
  TSAlpha2GetParams - gets the algorithmic parameters for TSALPHA2

  Not Collective

  Input Parameter:
+  ts - timestepping context
.  \alpha_m - algorithmic parameter
.  \alpha_f - algorithmic parameter
.  \gamma   - algorithmic parameter
-  \beta    - algorithmic parameter

  Note:
  Use of this function is normally only required to hack TSGALPHA to
  use a modified integration scheme. Users should call
  TSAlpha2SetRadius() to set the high-frequency damping (i.e. spectral
  radius of the method) in order so select optimal values for these
  parameters.

  Level: advanced

.seealso: TSAlpha2SetRadius(), TSAlpha2SetParams()
@*/
PetscErrorCode TSAlpha2GetParams(TS ts,PetscReal *alpha_m,PetscReal *alpha_f,PetscReal *gamma,PetscReal *beta)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (alpha_m) PetscValidPointer(alpha_m,2);
  if (alpha_f) PetscValidPointer(alpha_f,3);
  if (gamma)   PetscValidPointer(gamma,4);
  if (beta)    PetscValidPointer(beta,5);
  ierr = PetscUseMethod(ts,"TSAlpha2GetParams_C",(TS,PetscReal*,PetscReal*,PetscReal*,PetscReal*),(ts,alpha_m,alpha_f,gamma,beta));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
