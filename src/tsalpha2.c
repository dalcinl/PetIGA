/*
  Code for timestepping with implicit generalized-\alpha method
  for second order systems.
*/
#include <petscts2.h>
#if PETSC_VERSION_LT(3,6,0)
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/
#else
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#endif

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
  "@article{Chung1993,\n"
  "  title   = {A Time Integration Algorithm for Structural Dynamics with Improved Numerical Dissipation: The Generalized-$\\alpha$ Method},\n"
  "  author  = {J. Chung, G. M. Hubert},\n"
  "  journal = {ASME Journal of Applied Mechanics},\n"
  "  volume  = {60},\n"
  "  number  = {2},\n"
  "  pages   = {371--375},\n"
  "  year    = {1993},\n"
  "  issn    = {0021-8936},\n"
  "  doi     = {http://dx.doi.org/10.1115/1.2900803}\n}\n";

#if PETSC_VERSION_LT(3,5,0)
#define PetscCitationsRegister(a,b) ((void)a,(void)b,0)
#define TSPostStage(ts,t,n,x) 0
static PetscErrorCode TSRollBack_Alpha(TS);
#define TSRollBack(ts) \
  TSRollBack_Alpha(ts); \
  ts->ptime -= next_time_step; \
  ts->time_step = next_time_step;
#endif

#if PETSC_VERSION_LT(3,7,0)
#define TSAdaptCheckStage(adapt,ts,t,X,accept) TSAdaptCheckStage(adapt,ts,accept)
#endif

typedef struct {

  PetscReal stage_time;
  PetscReal shift_V;
  PetscReal shift_A;
  PetscReal scale_F;
  Vec       vec_dot;
  Vec       X0,Xa,X1;
  Vec       V0,Va,V1;
  Vec       A0,Aa,A1;
  Vec       vec_sol_prev;
  Vec       vec_dot_prev;
  Vec       work;

  PetscReal Alpha_m;
  PetscReal Alpha_f;
  PetscReal Gamma;
  PetscReal Beta;

  PetscBool    adapt;
  PetscInt     order;
  TSStepStatus status;

} TS_Alpha;

#undef __FUNCT__
#define __FUNCT__ "TSAlpha_StageTime"
static PetscErrorCode TSAlpha_StageTime(TS ts)
{
  TS_Alpha  *th = (TS_Alpha*)ts->data;
  PetscReal t  = ts->ptime;
  PetscReal dt = ts->time_step;
  PetscReal Alpha_m = th->Alpha_m;
  PetscReal Alpha_f = th->Alpha_f;
  PetscReal Gamma   = th->Gamma;
  PetscReal Beta    = th->Beta;

  PetscFunctionBegin;
  th->stage_time = t + Alpha_f*dt;
  th->shift_V = Gamma/(dt*Beta);
  th->shift_A = Alpha_m/(Alpha_f*dt*dt*Beta);
  th->scale_F = 1/Alpha_f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlpha_StageVecs"
static PetscErrorCode TSAlpha_StageVecs(TS ts,Vec X)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  Vec            X1 = X,      V1 = th->V1, A1 = th->A1;
  Vec            Xa = th->Xa, Va = th->Va, Aa = th->Aa;
  Vec            X0 = th->X0, V0 = th->V0, A0 = th->A0;
  PetscReal      dt = ts->time_step;
  PetscReal      Alpha_m = th->Alpha_m;
  PetscReal      Alpha_f = th->Alpha_f;
  PetscReal      Gamma   = th->Gamma;
  PetscReal      Beta    = th->Beta;
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
#define __FUNCT__ "TS_SNESSolve"
static PetscErrorCode TS_SNESSolve(TS ts,Vec b,Vec x)
{
  PetscInt       nits,lits;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESSolve(ts->snes,b,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(ts->snes,&nits);CHKERRQ(ierr);
  ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
  ts->snes_its += nits; ts->ksp_its += lits;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlpha_InitStep"
static PetscErrorCode TSAlpha_InitStep(TS ts,PetscBool *initok)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      time_step;
  PetscReal      alpha_m,alpha_f,gamma,beta;
  Vec            X0 = ts->vec_sol, X1, X2 = th->X1;
  Vec            V0 = th->vec_dot, V1, V2 = th->V1;
  PetscBool      stageok;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(X0,&X1);CHKERRQ(ierr);
  ierr = VecDuplicate(X0,&V1);CHKERRQ(ierr);
  ierr = TSAlpha2GetParams(ts,&alpha_m,&alpha_f,&gamma,&beta);CHKERRQ(ierr);
  ierr = TSAlpha2SetParams(ts,1,1,1,0.5);CHKERRQ(ierr);

  ierr = TSGetTimeStep(ts,&time_step);CHKERRQ(ierr);
  ts->time_step = time_step/2;
  ierr = TSAlpha_StageTime(ts);CHKERRQ(ierr);
  th->stage_time = ts->ptime;

  th->stage_time += ts->time_step;
  ierr = VecCopy(X0,th->X0);CHKERRQ(ierr);
  ierr = VecCopy(V0,th->V0);CHKERRQ(ierr);
  ierr = TSPreStage(ts,th->stage_time);CHKERRQ(ierr);
  ierr = VecCopy(th->X0,X1);CHKERRQ(ierr);
  ierr = TS_SNESSolve(ts,NULL,X1);CHKERRQ(ierr);
  ierr = VecCopy(th->V1,V1);CHKERRQ(ierr);
  ierr = TSPostStage(ts,th->stage_time,0,&X1);CHKERRQ(ierr);
  ierr = TSAdaptCheckStage(ts->adapt,ts,th->stage_time,X1,&stageok);CHKERRQ(ierr);
  if (!stageok) goto finally;

  th->stage_time += ts->time_step;
  ierr = VecCopy(X1,th->X0);CHKERRQ(ierr);
  ierr = VecCopy(V1,th->V0);CHKERRQ(ierr);
  ierr = TSPreStage(ts,th->stage_time);CHKERRQ(ierr);
  ierr = VecCopy(th->X0,X2);CHKERRQ(ierr);
  ierr = TS_SNESSolve(ts,NULL,X2);CHKERRQ(ierr);
  ierr = VecCopy(th->V1,V2);CHKERRQ(ierr);
  ierr = TSPostStage(ts,th->stage_time,0,&X2);CHKERRQ(ierr);
  ierr = TSAdaptCheckStage(ts->adapt,ts,th->stage_time,X1,&stageok);CHKERRQ(ierr);
  if (!stageok) goto finally;

  ierr = TSSetTimeStep(ts,time_step);CHKERRQ(ierr);
  ierr = VecZeroEntries(th->A0);CHKERRQ(ierr);
  ierr = VecAXPY(th->A0,-3/ts->time_step,V0);CHKERRQ(ierr);
  ierr = VecAXPY(th->A0,+4/ts->time_step,V1);CHKERRQ(ierr);
  ierr = VecAXPY(th->A0,-1/ts->time_step,V2);CHKERRQ(ierr);
  if (th->vec_sol_prev) {
    ierr = VecZeroEntries(th->vec_sol_prev);CHKERRQ(ierr);
    ierr = VecAXPY(th->vec_sol_prev,+2,X2);CHKERRQ(ierr);
    ierr = VecAXPY(th->vec_sol_prev,-4,X1);CHKERRQ(ierr);
    ierr = VecAXPY(th->vec_sol_prev,+2,X0);CHKERRQ(ierr);
  }
  if (th->vec_dot_prev) {
    ierr = VecZeroEntries(th->vec_dot_prev);CHKERRQ(ierr);
    ierr = VecAXPY(th->vec_dot_prev,+2,V2);CHKERRQ(ierr);
    ierr = VecAXPY(th->vec_dot_prev,-4,V1);CHKERRQ(ierr);
    ierr = VecAXPY(th->vec_dot_prev,+2,V0);CHKERRQ(ierr);
  }

 finally:
  if (initok) *initok = stageok;
  ierr = TSAlpha2SetParams(ts,alpha_m,alpha_f,gamma,beta);CHKERRQ(ierr);
  ierr = VecCopy(ts->vec_sol,th->X0);CHKERRQ(ierr);
  ierr = VecCopy(th->vec_dot,th->V0);CHKERRQ(ierr);
  ierr = VecDestroy(&X1);CHKERRQ(ierr);
  ierr = VecDestroy(&V1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSStep_Alpha"
static PetscErrorCode TSStep_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscBool      stageok,accept = PETSC_TRUE;
  PetscInt       next_scheme,rejections = 0;
  PetscReal      next_time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);

  ierr = VecCopy(ts->vec_sol,th->X0);CHKERRQ(ierr);
  ierr = VecCopy(th->vec_dot,th->V0);CHKERRQ(ierr);
  ierr = VecCopy(th->A1,th->A0);CHKERRQ(ierr);
  th->status = TS_STEP_INCOMPLETE;

  while (!ts->reason && th->status != TS_STEP_COMPLETE) {
    ierr = TSPreStep(ts);CHKERRQ(ierr);
    if (ts->steps == 0) {
      ierr = TSAlpha_InitStep(ts,&stageok);CHKERRQ(ierr);
      if (!stageok) {accept = PETSC_FALSE; goto reject_step;}
    }

    ierr = TSAlpha_StageTime(ts);CHKERRQ(ierr);
    ierr = VecCopy(th->X0,th->X1);CHKERRQ(ierr);
    ierr = TSPreStage(ts,th->stage_time);CHKERRQ(ierr);
    ierr = TS_SNESSolve(ts,NULL,th->X1);CHKERRQ(ierr);
    ierr = TSPostStage(ts,th->stage_time,0,&th->X1);CHKERRQ(ierr);
    ierr = TSAdaptCheckStage(ts->adapt,ts,th->stage_time,th->X1,&stageok);CHKERRQ(ierr);
    if (!stageok) {accept = PETSC_FALSE; goto reject_step;}

    ierr = TSEvaluateStep2(ts,th->order,ts->vec_sol,th->vec_dot,NULL);CHKERRQ(ierr);
    th->status = TS_STEP_PENDING;

    ierr = TSAdaptCandidatesClear(ts->adapt);CHKERRQ(ierr);
    ierr = TSAdaptCandidateAdd(ts->adapt,"",th->order,1,1.0,1.0,PETSC_TRUE);CHKERRQ(ierr);
    ierr = TSAdaptChoose(ts->adapt,ts,ts->time_step,&next_scheme,&next_time_step,&accept);CHKERRQ(ierr);
    if (!accept) {
      ts->ptime += next_time_step;
      ierr = TSRollBack(ts);CHKERRQ(ierr);
      th->status = TS_STEP_INCOMPLETE;
      goto reject_step;
    }

    ts->ptime += ts->time_step;
    ts->time_step = next_time_step;
    ts->steps++;
    th->status = TS_STEP_COMPLETE;
    break;

  reject_step:
    ts->reject++;
    if (!ts->reason && ++rejections > ts->max_reject && ts->max_reject >= 0) {
      ts->reason = TS_DIVERGED_STEP_REJECTED;
      ierr = PetscInfo2(ts,"Step=%D, step rejections %D greater than current TS allowed, stopping solve\n",ts->steps,rejections);CHKERRQ(ierr);
    }
  }

  if (th->vec_sol_prev && !ts->reason) {ierr = VecCopy(th->X0,th->vec_sol_prev);CHKERRQ(ierr);}
  if (th->vec_dot_prev && !ts->reason) {ierr = VecCopy(th->V0,th->vec_dot_prev);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSEvaluateStep_Alpha"
static PetscErrorCode TSEvaluateStep_Alpha(TS ts,PetscInt order,Vec U,PETSC_UNUSED PetscBool *done)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (order == 0) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"No time-step adaptivity implemented for 1st order alpha method; Run with -ts_adapt_type none");
  if (!th->work) {ierr = VecDuplicate(ts->vec_sol,&th->work);CHKERRQ(ierr);}
  ierr = TSEvaluateStep2(ts,order,U,th->work,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRollBack_Alpha"
static PetscErrorCode TSRollBack_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(th->X0,ts->vec_sol);CHKERRQ(ierr);
  ierr = VecCopy(th->V0,th->vec_dot);CHKERRQ(ierr);
  th->status = TS_STEP_INCOMPLETE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSInterpolate_Alpha"
static PetscErrorCode TSInterpolate_Alpha(TS ts,PetscReal t,Vec X)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!th->work) {ierr = VecDuplicate(ts->vec_sol,&th->work);CHKERRQ(ierr);}
  ierr = TSInterpolate2(ts,t,X,th->work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormFunction_Alpha"
static PetscErrorCode SNESTSFormFunction_Alpha(PETSC_UNUSED SNES snes,Vec X,Vec F,TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      ta = th->stage_time;
  Vec            Xa = th->Xa, Va = th->Va, Aa = th->Aa;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSAlpha_StageVecs(ts,X);CHKERRQ(ierr);
  /* F = Function(ta,Xa,Va,Aa) */
  ierr = TSComputeIFunction2(ts,ta,Xa,Va,Aa,F,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecScale(F,th->scale_F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormJacobian_Alpha"
static PetscErrorCode SNESTSFormJacobian_Alpha(PETSC_UNUSED SNES snes,
                                               PETSC_UNUSED Vec X,
#if PETSC_VERSION_LT(3,5,0)
                                               Mat *J,Mat *P,MatStructure *m,
#else
                                               Mat J,Mat P,
#endif
                                               TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      ta = th->stage_time;
  Vec            Xa = th->Xa, Va = th->Va, Aa = th->Aa;
  PetscReal      dVdX = th->shift_V, dAdX = th->shift_A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* J,P = Jacobian(ta,Xa,Va,Aa) */
#if PETSC_VERSION_LT(3,5,0)
  *m = SAME_NONZERO_PATTERN;
  ierr = TSComputeIJacobian2(ts,ta,Xa,Va,Aa,dVdX,dAdX,*J,*P,PETSC_FALSE);CHKERRQ(ierr);
#else
  ierr = TSComputeIJacobian2(ts,ta,Xa,Va,Aa,dVdX,dAdX,J,P,PETSC_FALSE);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSReset_Alpha"
static PetscErrorCode TSReset_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&th->vec_dot);CHKERRQ(ierr);
  ierr = VecDestroy(&th->X0);CHKERRQ(ierr);
  ierr = VecDestroy(&th->Xa);CHKERRQ(ierr);
  ierr = VecDestroy(&th->X1);CHKERRQ(ierr);
  ierr = VecDestroy(&th->V0);CHKERRQ(ierr);
  ierr = VecDestroy(&th->Va);CHKERRQ(ierr);
  ierr = VecDestroy(&th->V1);CHKERRQ(ierr);
  ierr = VecDestroy(&th->A0);CHKERRQ(ierr);
  ierr = VecDestroy(&th->Aa);CHKERRQ(ierr);
  ierr = VecDestroy(&th->A1);CHKERRQ(ierr);
  ierr = VecDestroy(&th->vec_sol_prev);CHKERRQ(ierr);
  ierr = VecDestroy(&th->vec_dot_prev);CHKERRQ(ierr);
  ierr = VecDestroy(&th->work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_Alpha"
static PetscErrorCode TSDestroy_Alpha(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_Alpha(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSSetSolution2_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSGetSolution2_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSSolve2_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSInterpolate2_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSEvaluateStep2_C",NULL);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlpha2UseAdapt_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlpha2SetRadius_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlpha2SetParams_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlpha2GetParams_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscBool always_accept;      /* always accept step */
  PetscReal clip[2];            /* admissible decrease/increase factors */
  PetscReal safety;             /* safety factor relative to target error */
  PetscReal reject_safety;      /* extra safety factor if the last step was rejected */
  Vec       Y;
} TSAdapt_Basic;

#if PETSC_VERSION_LT(3,6,0)
#define TSErrorWeightedNorm(ts,X,Y,wntype,norm) 0; do {    \
    Vec            _save = ts->vec_sol;                    \
    PetscErrorCode _ierr_1;                                \
    ts->vec_sol = X;                                       \
    _ierr_1 = TSErrorNormWRMS(ts,Y,norm);CHKERRQ(_ierr_1); \
    ts->vec_sol = _save;                                   \
  } while (0)
#endif

#undef __FUNCT__
#define __FUNCT__ "TSAdaptChoose_Alpha"
static PetscErrorCode TSAdaptChoose_Alpha(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte)
{
  TSAdapt_Basic  *basic = (TSAdapt_Basic*)adapt->data;
  PetscInt       order = adapt->candidates.order[0];
  PetscReal      enorm,hfac_lte,h_lte,safety;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  {
    TS_Alpha  *th = (TS_Alpha*)ts->data;
    PetscReal enormX,enormV;
    if (!basic->Y) {ierr = VecDuplicate(ts->vec_sol,&basic->Y);CHKERRQ(ierr);}
    if (!th->work) {ierr = VecDuplicate(ts->vec_sol,&th->work);CHKERRQ(ierr);}
    ierr = TSEvaluateStep2(ts,order-1,basic->Y,th->work,NULL);CHKERRQ(ierr);
    ierr = TSErrorWeightedNorm(ts,th->X1,basic->Y,adapt->wnormtype,&enormX);CHKERRQ(ierr);
    ierr = TSErrorWeightedNorm(ts,th->V1,th->work,adapt->wnormtype,&enormV);CHKERRQ(ierr);
    enorm = PetscSqrtReal(PetscSqr(enormX)/2 + PetscSqr(enormV)/2);
  }

  safety = basic->safety;
  if (enorm > 1.0) {
    if (!*accept) safety *= basic->reject_safety; /* The last attempt also failed, shorten more aggressively */
    if (h < (1 + PETSC_SQRT_MACHINE_EPSILON)*adapt->dt_min) {
      ierr    = PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting because step size %g is at minimum\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else if (basic->always_accept) {
      ierr    = PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting step of size %g because always_accept is set\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_TRUE;
    } else {
      ierr    = PetscInfo2(adapt,"Estimated scaled local truncation error %g, rejecting step of size %g\n",(double)enorm,(double)h);CHKERRQ(ierr);
      *accept = PETSC_FALSE;
    }
  } else {
    ierr    = PetscInfo2(adapt,"Estimated scaled local truncation error %g, accepting step of size %g\n",(double)enorm,(double)h);CHKERRQ(ierr);
    *accept = PETSC_TRUE;
  }

  /* The optimal new step based purely on local truncation error for this step. */
  hfac_lte = safety * PetscRealPart(PetscPowScalar((PetscScalar)enorm,(PetscReal)(-1./order)));
  h_lte    = h * PetscClipInterval(hfac_lte,basic->clip[0],basic->clip[1]);

  *next_sc = 0;
  *next_h  = PetscClipInterval(h_lte,adapt->dt_min,adapt->dt_max);
  *wlte    = enorm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_Alpha"
static PetscErrorCode TSSetUp_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!th->vec_dot) {ierr = VecDuplicate(ts->vec_sol,&th->vec_dot);CHKERRQ(ierr);}
  ierr = VecDuplicate(ts->vec_sol,&th->X0);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Xa);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->X1);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->V0);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Va);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->V1);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->A0);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Aa);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->A1);CHKERRQ(ierr);
  ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
  if (!th->adapt) {
    ierr = TSAdaptDestroy(&ts->adapt);CHKERRQ(ierr);
    ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
    ierr = TSAdaptSetType(ts->adapt,TSADAPTNONE);CHKERRQ(ierr);
  } else {
    PetscBool match;
    ierr = PetscObjectTypeCompare((PetscObject)ts->adapt,TSADAPTBASIC,&match);CHKERRQ(ierr);
    if (match) {ts->adapt->ops->choose = TSAdaptChoose_Alpha;}
    ierr = VecDuplicate(ts->vec_sol,&th->vec_sol_prev);CHKERRQ(ierr);
    ierr = VecDuplicate(ts->vec_sol,&th->vec_dot_prev);CHKERRQ(ierr);
  }
  ierr = TSGetSNES(ts,&ts->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if PETSC_VERSION_LT(3,6,0)
#define PetscOptionsHead(obj,head) ((void)(obj),PetscOptionsHead(head))
#endif
#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_Alpha"
static PetscErrorCode TSSetFromOptions_Alpha(PetscOptions *PetscOptionsObject,TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Generalized-Alpha ODE solver options");CHKERRQ(ierr);
  {
    PetscBool flg;
    PetscReal radius = 1;
    PetscBool adapt  = th->adapt;
    ierr = PetscOptionsReal("-ts_alpha_radius","Spectral radius (high-frequency dissipation)","TSAlpha2SetRadius",radius,&radius,&flg);CHKERRQ(ierr);
    if (flg) {ierr = TSAlpha2SetRadius(ts,radius);CHKERRQ(ierr);}
    ierr = PetscOptionsReal("-ts_alpha_alpha_m","Algoritmic parameter alpha_m","TSAlpha2SetParams",th->Alpha_m,&th->Alpha_m,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_alpha_f","Algoritmic parameter alpha_f","TSAlpha2SetParams",th->Alpha_f,&th->Alpha_f,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_gamma","Algoritmic parameter gamma","TSAlpha2SetParams",th->Gamma,&th->Gamma,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_beta","Algoritmic parameter beta","TSAlpha2SetParams",th->Beta,&th->Beta,NULL);CHKERRQ(ierr);
    ierr = TSAlpha2SetParams(ts,th->Alpha_m,th->Alpha_f,th->Gamma,th->Beta);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ts_alpha_adapt","Use time-step adaptivity with the Alpha method","TSAlpha2UseAdapt",adapt,&adapt,&flg);CHKERRQ(ierr);
    if (flg) {ierr = TSAlpha2UseAdapt(ts,adapt);CHKERRQ(ierr);}
#if PETSC_VERSION_LT(3,6,0)
    ierr = SNESSetFromOptions(ts->snes);CHKERRQ(ierr);
#endif
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#if PETSC_VERSION_LT(3,6,0)
static PetscErrorCode TSSetFromOptions_Alpha_Legacy(TS ts) {return TSSetFromOptions_Alpha(NULL,ts);}
#define TSSetFromOptions_Alpha TSSetFromOptions_Alpha_Legacy
#endif

#undef __FUNCT__
#define __FUNCT__ "TSView_Alpha"
static PetscErrorCode TSView_Alpha(TS ts,PetscViewer viewer)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscBool      ascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&ascii);CHKERRQ(ierr);
  if (ascii)    {ierr = PetscViewerASCIIPrintf(viewer,"  Alpha_m=%g, Alpha_f=%g, Gamma=%g, Beta=%g\n",(double)th->Alpha_m,(double)th->Alpha_f,(double)th->Gamma,(double)th->Beta);CHKERRQ(ierr);}
  if (ts->snes) {ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

#undef __FUNCT__
#define __FUNCT__ "TSSetSolution2_Alpha"
static PetscErrorCode TSSetSolution2_Alpha(TS ts,Vec X,Vec V)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)V);CHKERRQ(ierr);
  ierr = VecDestroy(&th->vec_dot);CHKERRQ(ierr);
  th->vec_dot = V;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetSolution2_Alpha"
static PetscErrorCode TSGetSolution2_Alpha(TS ts,Vec *X, Vec *V)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!th->vec_dot && ts->vec_sol) {
    ierr = VecDuplicate(ts->vec_sol,&th->vec_dot);CHKERRQ(ierr);
  }
  if (X) {ierr = TSGetSolution(ts,X);CHKERRQ(ierr);}
  if (V) {*V = th->vec_dot;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSolve2_Alpha"
static PetscErrorCode TSSolve2_Alpha(TS ts,Vec X,Vec V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSSetSolution2(ts,X,V);CHKERRQ(ierr);
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSInterpolate2_Alpha"
static PetscErrorCode TSInterpolate2_Alpha(TS ts,PetscReal t,Vec X,Vec V)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      dt  = ts->ptime - t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(th->vec_dot,V);CHKERRQ(ierr);
  ierr = VecAXPY(V,-dt*(1-th->Gamma),th->A0);CHKERRQ(ierr);
  ierr = VecAXPY(V,-dt*th->Gamma,th->A1);CHKERRQ(ierr);
  ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
  ierr = VecAXPY(X,-dt,V);CHKERRQ(ierr);
  ierr = VecAXPY(X,-dt*dt*((PetscReal)0.5-th->Beta),th->A0);CHKERRQ(ierr);
  ierr = VecAXPY(X,-dt*dt*th->Beta,th->A1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSEvaluateStep2_Alpha"
static PetscErrorCode TSEvaluateStep2_Alpha(TS ts,PetscInt order,Vec X,Vec V,PETSC_UNUSED PetscBool *done)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (order == 0) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"No time-step adaptivity implemented for 1st order alpha method; Run with -ts_adapt_type none");
  if (order == th->order) {
    ierr = VecCopy(th->X1,X);CHKERRQ(ierr);
    ierr = VecCopy(th->V1,V);CHKERRQ(ierr);
  } else if (order == th->order-1) {
    if (ts->steps > 0) {
      PetscReal a = 1 + ts->time_step_prev/ts->time_step;
      PetscScalar scal[3]; Vec vecX[3],vecV[3];
      scal[0] = +1/a;   scal[1] = -1/(a-1); scal[2] = +1/(a*(a-1));
      vecX[0] = th->X1; vecX[1] = th->X0;   vecX[2] = th->vec_sol_prev;
      vecV[0] = th->V1; vecV[1] = th->V0;   vecV[2] = th->vec_dot_prev;
      ierr = VecCopy(th->X1,X);CHKERRQ(ierr);
      ierr = VecMAXPY(X,3,scal,vecX);CHKERRQ(ierr);
      ierr = VecCopy(th->V1,V);CHKERRQ(ierr);
      ierr = VecMAXPY(V,3,scal,vecV);CHKERRQ(ierr);
    } else {
      ierr = VecWAXPY(X,1.0,th->vec_sol_prev,th->X1);CHKERRQ(ierr);
      ierr = VecWAXPY(V,1.0,th->vec_dot_prev,th->V1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

#undef __FUNCT__
#define __FUNCT__ "TSAlpha2UseAdapt_Alpha"
static PetscErrorCode TSAlpha2UseAdapt_Alpha(TS ts,PetscBool use)
{
  TS_Alpha *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  if (use == th->adapt) PetscFunctionReturn(0);
  if (ts->setupcalled) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ORDER,"Cannot change adaptivity after TSSetUp()");
  th->adapt = use;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlpha2SetRadius_Alpha"
static PetscErrorCode TSAlpha2SetRadius_Alpha(TS ts,PetscReal radius)
{
  PetscReal      alpha_m,alpha_f,gamma,beta;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (radius < 0 || radius > 1) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Radius %g not in range [0,1]",(double)radius);
  alpha_m = (2-radius)/(1+radius);
  alpha_f = 1/(1+radius);
  gamma   = (PetscReal)0.5 + alpha_m - alpha_f;
  beta    = (PetscReal)0.5 * (1 + alpha_m - alpha_f); beta *= beta;
  ierr = TSAlpha2SetParams(ts,alpha_m,alpha_f,gamma,beta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlpha2SetParams_Alpha"
static PetscErrorCode TSAlpha2SetParams_Alpha(TS ts,PetscReal alpha_m,PetscReal alpha_f,PetscReal gamma,PetscReal beta)
{
  TS_Alpha  *th = (TS_Alpha*)ts->data;
  PetscReal tol = 100*PETSC_MACHINE_EPSILON;
  PetscReal res = ((PetscReal)0.5 + alpha_m - alpha_f) - gamma;

  PetscFunctionBegin;
  th->Alpha_m = alpha_m;
  th->Alpha_f = alpha_f;
  th->Gamma   = gamma;
  th->Beta    = beta;
  th->order   = (PetscAbsReal(res) < tol) ? 2 : 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlpha2GetParams_Alpha"
static PetscErrorCode TSAlpha2GetParams_Alpha(TS ts,PetscReal *alpha_m,PetscReal *alpha_f,PetscReal *gamma,PetscReal *beta)
{
  TS_Alpha *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  if (alpha_m) *alpha_m = th->Alpha_m;
  if (alpha_f) *alpha_f = th->Alpha_f;
  if (gamma)   *gamma   = th->Gamma;
  if (beta)    *beta    = th->Beta;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

/*MC
      TSALPHA2 - DAE solver using the implicit Generalized-Alpha method
                 for second-order systems

  Level: beginner

  References:
  J. Chung, G.M.Hubert. "A Time Integration Algorithm for Structural
  Dynamics with Improved Numerical Dissipation: The Generalized-alpha
  Method" ASME Journal of Applied Mechanics, 60, 371:375, 1993.

.seealso:  TS, TSCreate(), TSSetType()
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSCreate_Alpha2"
PetscErrorCode TSCreate_Alpha2(TS ts);
PetscErrorCode TSCreate_Alpha2(TS ts)
{
  TS_Alpha       *th;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_Alpha;
  ts->ops->destroy        = TSDestroy_Alpha;
  ts->ops->view           = TSView_Alpha;
  ts->ops->setup          = TSSetUp_Alpha;
  ts->ops->setfromoptions = TSSetFromOptions_Alpha;
  ts->ops->step           = TSStep_Alpha;
  ts->ops->evaluatestep   = TSEvaluateStep_Alpha;
#if PETSC_VERSION_GE(3,5,0)
  ts->ops->rollback       = TSRollBack_Alpha;
#endif
  ts->ops->interpolate    = TSInterpolate_Alpha;
  ts->ops->snesfunction   = SNESTSFormFunction_Alpha;
  ts->ops->snesjacobian   = SNESTSFormJacobian_Alpha;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSSetSolution2_C",TSSetSolution2_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSGetSolution2_C",TSGetSolution2_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSSolve2_C",TSSolve2_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSInterpolate2_C",TSInterpolate2_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSEvaluateStep2_C",TSEvaluateStep2_Alpha);CHKERRQ(ierr);

#if PETSC_VERSION_LT(3,5,0)
  ierr = PetscNewLog(ts,TS_Alpha,&th);CHKERRQ(ierr);
#else
  ierr = PetscNewLog(ts,&th);CHKERRQ(ierr);
#endif
  ts->data = (void*)th;

  th->Alpha_m = 0.5;
  th->Alpha_f = 0.5;
  th->Gamma   = 0.5;
  th->Beta    = 0.25;

  th->adapt = PETSC_FALSE;
  th->order = 2;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlpha2UseAdapt_C",TSAlpha2UseAdapt_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlpha2SetRadius_C",TSAlpha2SetRadius_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlpha2SetParams_C",TSAlpha2SetParams_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlpha2GetParams_C",TSAlpha2GetParams_Alpha);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ------------------------------------------------------------ */

#undef __FUNCT__
#define __FUNCT__ "DMTSDuplicate"
static PetscErrorCode DMTSDuplicate(DMTS oldtsdm,DMTS newtsdm)
{
  PetscObject    oldobj = (PetscObject)oldtsdm;
  PetscObject    newobj = (PetscObject)newtsdm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFunctionListDuplicate(oldobj->qlist,&newobj->qlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSSetIFunction2"
PetscErrorCode DMTSSetIFunction2(DM dm,TSIFunction2 fun,void *ctx)
{
  DMTS           tsdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&tsdm);CHKERRQ(ierr);
  tsdm->ops->duplicate = DMTSDuplicate;
  if (fun) {ierr = PetscObjectComposeFunction((PetscObject)tsdm,"TSIFunction2_C",fun);CHKERRQ(ierr);}
  if (ctx) {ierr = DMTSSetIFunction(dm,NULL,ctx);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSSetIJacobian2"
PetscErrorCode DMTSSetIJacobian2(DM dm,TSIJacobian2 jac,void *ctx)
{
  DMTS           tsdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTSWrite(dm,&tsdm);CHKERRQ(ierr);
  tsdm->ops->duplicate = DMTSDuplicate;
  if (jac) {ierr = PetscObjectComposeFunction((PetscObject)tsdm,"TSIJacobian2_C",jac);CHKERRQ(ierr);}
  if (ctx) {ierr = DMTSSetIJacobian(dm,NULL,ctx);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSGetIFunction2"
PetscErrorCode DMTSGetIFunction2(DM dm,TSIFunction2 *fun,void **ctx)
{
  DMTS           tsdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm,&tsdm);CHKERRQ(ierr);
  if (fun) {ierr = PetscObjectQueryFunction((PetscObject)tsdm,"TSIFunction2_C",(PetscVoidFunction*)fun);CHKERRQ(ierr);}
  if (ctx) {ierr = DMTSGetIFunction(dm,NULL,ctx);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSGetIJacobian2"
PetscErrorCode DMTSGetIJacobian2(DM dm,TSIJacobian2 *jac,void **ctx)
{
  DMTS           tsdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMTS(dm,&tsdm);CHKERRQ(ierr);
  if (jac) {ierr = PetscObjectQueryFunction((PetscObject)tsdm,"TSIJacobian2_C",(PetscVoidFunction*)jac);CHKERRQ(ierr);}
  if (ctx) {ierr = DMTSGetIJacobian(dm,NULL,ctx);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

#undef __FUNCT__
#define __FUNCT__ "TSSetIFunction2"
/*@C
   TSSetIFunction2 - Set the function to compute F(t,U,U_t,U_tt) where F = 0 is the DAE to be solved.

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

.seealso: TSALPHA2, TSSetIJacobian2()
@*/
PetscErrorCode TSSetIFunction2(TS ts,Vec F,TSIFunction2 fun,void *ctx)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (F) PetscValidHeaderSpecific(F,VEC_CLASSID,2);
  if (F) {ierr = TSSetIFunction(ts,F,NULL,NULL);CHKERRQ(ierr);}
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetIFunction2(dm,fun,ctx);CHKERRQ(ierr);
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

.seealso: TSALPHA2, TSSetIFunction2()
@*/
PetscErrorCode TSSetIJacobian2(TS ts,Mat J,Mat P,TSIJacobian2 jac,void *ctx)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (J) PetscValidHeaderSpecific(J,MAT_CLASSID,2);
  if (P) PetscValidHeaderSpecific(P,MAT_CLASSID,3);
  if (J || P) {ierr = TSSetIJacobian(ts,J,P,NULL,NULL);CHKERRQ(ierr);}
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetIJacobian2(dm,jac,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeIFunction2"
PetscErrorCode TSComputeIFunction2(TS ts,PetscReal t,Vec X,Vec V,Vec A,Vec F,PetscBool imex)
{
  DM             dm;
  TSIFunction2   IFunction2;
  void           *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(A,VEC_CLASSID,5);
  PetscValidHeaderSpecific(F,VEC_CLASSID,6);
  PetscValidLogicalCollectiveBool(ts,imex,7);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetIFunction2(dm,&IFunction2,&ctx);CHKERRQ(ierr);
  if (IFunction2) {
    PetscStackPush("TS user implicit function");
    ierr = IFunction2(ts,t,X,V,A,F,ctx);CHKERRQ(ierr);
    PetscStackPop;
  } else {
    ierr = TSComputeIFunction(ts,t,X,A,F,imex);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeIJacobian2"
PetscErrorCode TSComputeIJacobian2(TS ts,PetscReal t,Vec X,Vec V,Vec A,PetscReal shiftV,PetscReal shiftA,Mat J,Mat P,PetscBool imex)
{
  DM             dm;
  TSIJacobian2   IJacobian2;
  void           *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  PetscValidHeaderSpecific(A,VEC_CLASSID,5);
  PetscValidHeaderSpecific(J,MAT_CLASSID,8);
  PetscValidHeaderSpecific(P,MAT_CLASSID,9);
  PetscValidLogicalCollectiveBool(ts,imex,7);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetIJacobian2(dm,&IJacobian2,&ctx);CHKERRQ(ierr);
  if (IJacobian2) {
    PetscStackPush("TS user implicit Jacobian");
    ierr = IJacobian2(ts,t,X,V,A,shiftV,shiftA,J,P,ctx);CHKERRQ(ierr);
    PetscStackPop;
  } else {
#if PETSC_VERSION_LT(3,5,0)
    MatStructure m;
    ierr = TSComputeIJacobian(ts,t,X,A,shiftA,&J,&P,&m,imex);CHKERRQ(ierr);
#else
    ierr = TSComputeIJacobian(ts,t,X,A,shiftA,J,P,imex);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetSolution2"
/*@
   TSSetSolution2 - Sets the initial solution and time-derivative vectors
   for use by the TSALPHA2 routines.

   Logically Collective on TS and Vec

   Input Parameters:
+  ts - the TS context
.  X - the solution vector
-  V - the time-derivative vector

   Level: beginner

.keywords: TS, TSALPHA2, timestep, set, solution, initial conditions
@*/
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
/*@
   TSGetSolution2 - Returns the solution and time-derivative vectors
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
/*@
   TSSolve2 - Steps the requested number of timesteps.

   Collective on TS

   Input Parameter:
+  ts - the TS context
.  X - the solution vector (can be null if TSSetSolution2() was used, otherwise must contain the initial conditions)
-  V - the time-derivative vector (can be null if TSSetSolution2() was used, otherwise must contain the initial conditions)

   Level: beginner

.keywords: TS, TSALPHA2, timestep, solve

.seealso: TSALPHA2, TSSetSolution2()
@*/
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

#undef __FUNCT__
#define __FUNCT__ "TSInterpolate2"
/*@
   TSInterpolate2 - Interpolate the solution computed during the previous step to an arbitrary location in the interval

   Collective on TS

   Input Argument:
+  ts - time stepping context
-  t - time to interpolate to

   Output Argument:
+  X - solution at given time
-  V - time-derivative at given time

.keywords: TS, TSALPHA2, timestep, solve

.seealso: TSALPHA2, TSSolve2(), TSGetSolution2()
@*/
PetscErrorCode TSInterpolate2(TS ts,PetscReal t,Vec X,Vec V)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  if (t < ts->ptime - ts->time_step_prev || t > ts->ptime) SETERRQ3(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Requested time %g not in last time steps [%g,%g]",t,(double)(ts->ptime-ts->time_step_prev),(double)ts->ptime);
  ierr = PetscUseMethod(ts,"TSInterpolate2_C",(TS,PetscReal,Vec,Vec),(ts,t,X,V));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSEvaluateStep2"
/*@
   TSEvaluateStep - Evaluate the solution at the end of a time step with a given order of accuracy.

   Collective on TS

   Input Arguments:
+  ts - time stepping context
.  order - desired order of accuracy
-  done - whether the step was evaluated at this order (pass NULL to generate an error if not available)

   Output Arguments:
+  X - solution at the end of the current step
-  V - derivative at the end of the current step

   Level: advanced

   Notes:
   This function cannot be called until all stages have been evaluated.
   It is normally called by adaptive controllers before a step has been accepted and may also be called by the user after TSStep() has returned.

.seealso: TSStep(), TSAdapt
@*/
PetscErrorCode TSEvaluateStep2(TS ts,PetscInt order,Vec X,Vec V,PetscBool *done)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,3);
  PetscValidHeaderSpecific(V,VEC_CLASSID,4);
  ierr = PetscUseMethod(ts,"TSEvaluateStep2_C",(TS,PetscInt,Vec,Vec,PetscBool*),(ts,order,X,V,done));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

#undef __FUNCT__
#define __FUNCT__ "TSAlpha2UseAdapt"
/*@
  TSAlpha2UseAdapt - Use time-step adaptivity with the Alpha method

  Logically Collective on TS

  Input Parameter:
+  ts - timestepping context
-  use - flag to use adaptivity

  Options Database:
.  -ts_alpha_adapt

  Level: intermediate

.seealso: TSAdapt
@*/
PetscErrorCode TSAlpha2UseAdapt(TS ts,PetscBool use)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveBool(ts,use,2);
  ierr = PetscTryMethod(ts,"TSAlpha2UseAdapt_C",(TS,PetscBool),(ts,use));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  if (radius < 0 || radius > 1) SETERRQ1(((PetscObject)ts)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Radius %g not in range [0,1]",(double)radius);
  ierr = PetscTryMethod(ts,"TSAlpha2SetRadius_C",(TS,PetscReal),(ts,radius));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlpha2SetParams"
/*@
  TSAlpha2SetParams - sets the algorithmic parameters for TSALPHA2

  Logically Collective on TS

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
  Use of this function is normally only required to hack TSALPHA2 to
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
  PetscValidLogicalCollectiveReal(ts,gamma,4);
  PetscValidLogicalCollectiveReal(ts,beta,5);
  ierr = PetscTryMethod(ts,"TSAlpha2SetParams_C",(TS,PetscReal,PetscReal,PetscReal,PetscReal),(ts,alpha_m,alpha_f,gamma,beta));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlpha2GetParams"
/*@
  TSAlpha2GetParams - gets the algorithmic parameters for TSALPHA2

  Not Collective

  Input Parameter:
.  ts - timestepping context

  Output Parameters:
+  \alpha_m - algorithmic parameter
.  \alpha_f - algorithmic parameter
.  \gamma   - algorithmic parameter
-  \beta    - algorithmic parameter

  Note:
  Use of this function is normally only required to hack TSALPHA2 to
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
  if (alpha_m) PetscValidRealPointer(alpha_m,2);
  if (alpha_f) PetscValidRealPointer(alpha_f,3);
  if (gamma)   PetscValidRealPointer(gamma,4);
  if (beta)    PetscValidRealPointer(beta,5);
  ierr = PetscUseMethod(ts,"TSAlpha2GetParams_C",(TS,PetscReal*,PetscReal*,PetscReal*,PetscReal*),(ts,alpha_m,alpha_f,gamma,beta));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
