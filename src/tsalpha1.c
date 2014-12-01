/*
  Code for timestepping with implicit generalized-\alpha method
  for first order systems.
*/
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
  "@article{Jansen2000,\n"
  "  title   = {A generalized-$\\alpha$ method for integrating the filtered {N}avier--{S}tokes equations with a stabilized finite element method},\n"
  "  author  = {Kenneth E. Jansen and Christian H. Whiting and Gregory M. Hulbert},\n"
  "  journal = {Computer Methods in Applied Mechanics and Engineering},\n"
  "  volume  = {190},\n"
  "  number  = {3--4},\n"
  "  pages   = {305--319},\n"
  "  year    = {2000},\n"
  "  issn    = {0045-7825},\n"
  "  doi     = {http://dx.doi.org/10.1016/S0045-7825(00)00203-6}\n}\n";

#if PETSC_VERSION_LT(3,5,0)
#define PetscCitationsRegister(a,b) ((void)a,(void)b,0)
#define TSPostStage(ts,t,n,x) 0
static PetscErrorCode TSRollBack_Alpha(TS);
#define TSRollBack(ts) \
  TSRollBack_Alpha(ts); \
  ts->ptime -= next_time_step; \
  ts->time_step = next_time_step;
#endif

#if PETSC_VERSION_LT(3,4,0)
#define PetscObjectComm(o) ((o)->comm)
#define PetscObjectComposeFunction(o,n,f) \
        PetscObjectComposeFunction(o,n,"",(PetscVoidFunction)(f))
#endif

typedef struct {

  Vec X0,Xa,X1;
  Vec V0,Va,V1;
  Vec vec_sol_prev;

  PetscReal Alpha_m;
  PetscReal Alpha_f;
  PetscReal Gamma;

  PetscReal stage_time;
  PetscReal shift;

  PetscBool    adapt;
  PetscInt     order;
  TSStepStatus status;

} TS_Alpha;

#undef __FUNCT__
#define __FUNCT__ "TSStep_Alpha"
static PetscErrorCode TSStep_Alpha(TS ts)
{
  TS_Alpha            *th = (TS_Alpha*)ts->data;
  PetscInt            its,lits,reject,next_scheme;
  PetscReal           next_time_step;
  PetscBool           stageok,accept = PETSC_TRUE;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);

  if (th->vec_sol_prev) {ierr = VecCopy(ts->steps ? th->X0 : ts->vec_sol,th->vec_sol_prev);CHKERRQ(ierr);}

  next_time_step = ts->time_step;
  th->status = TS_STEP_INCOMPLETE;
  ierr = VecCopy(ts->vec_sol,th->X0);CHKERRQ(ierr);
  ierr = VecCopy(th->V1,th->V0);CHKERRQ(ierr);

  for (reject=0; reject<ts->max_reject && !ts->reason && th->status != TS_STEP_COMPLETE; reject++) {

    th->stage_time = ts->ptime + th->Alpha_f*ts->time_step;
    th->shift      = th->Alpha_m/(th->Alpha_f*th->Gamma*ts->time_step);
    ierr = TSPreStep(ts);CHKERRQ(ierr);

    ierr = TSPreStage(ts,th->stage_time);CHKERRQ(ierr);
    ierr = VecCopy(th->X0,th->X1);CHKERRQ(ierr);
    ierr = SNESSolve(ts->snes,NULL,th->X1);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
    ts->snes_its += its; ts->ksp_its += lits;
    ierr = TSPostStage(ts,th->stage_time,0,&th->X1);CHKERRQ(ierr);
    ierr = TSAdaptCheckStage(ts->adapt,ts,&stageok);CHKERRQ(ierr);
    if (!stageok) {accept = PETSC_FALSE; ts->reject++; continue;}

    ierr = TSEvaluateStep(ts,th->order,ts->vec_sol,NULL);CHKERRQ(ierr);
    th->status = TS_STEP_PENDING;

    ierr = TSAdaptCandidatesClear(ts->adapt);CHKERRQ(ierr);
    ierr = TSAdaptCandidateAdd(ts->adapt,"",th->order,1,1.0,1.0,PETSC_TRUE);CHKERRQ(ierr);
    ierr = TSAdaptChoose(ts->adapt,ts,ts->time_step,&next_scheme,&next_time_step,&accept);CHKERRQ(ierr);
    if (!accept) {
      ts->ptime += next_time_step;
      th->status = TS_STEP_INCOMPLETE;
      ierr = TSRollBack(ts);CHKERRQ(ierr);
      ts->reject++; continue;
    }

    ts->ptime += ts->time_step;
    ts->time_step = next_time_step;
    th->status = TS_STEP_COMPLETE;
    ts->steps++;
  }

  if (reject >= ts->max_reject) {
    ts->reason = TS_DIVERGED_STEP_REJECTED;
    ierr = PetscInfo3(ts,"Step=%D, step rejections %D greater than current TS allowed %D, stopping solve\n",ts->steps,reject,ts->max_reject);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSEvaluateStep_Alpha"
static PetscErrorCode TSEvaluateStep_Alpha(TS ts,PetscInt order,Vec U,PetscBool *done)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (order == 0) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"No time-step adaptivity implemented for 1st order alpha method; Run with -ts_adapt_type none");
  if (order == th->order) {
    ierr = VecCopy(th->X1,U);CHKERRQ(ierr);
  } else if (order && order == th->order-1) {
    PetscReal dt = ts->time_step;
    PetscReal dt_prev = ts->steps ? ts->time_step_prev : ts->time_step;
    PetscReal a = (dt+dt_prev)/dt;
    PetscScalar scals[3]; Vec vecs[3];
    scals[0] = (a+1)/a;   vecs[0] = th->X1;
    scals[1] = -a/(a-1);  vecs[1] = th->X0;
    scals[2] = 1/a/(a-1); vecs[2] = th->vec_sol_prev;
    ierr = VecCopy(th->X0,U);CHKERRQ(ierr);
    ierr = VecMAXPY(U,3,scals,vecs);CHKERRQ(ierr);
  }
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
  th->status = TS_STEP_INCOMPLETE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSInterpolate_Alpha"
static PetscErrorCode TSInterpolate_Alpha(TS ts,PetscReal t,Vec X)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscReal      dt  = t - ts->ptime;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
  ierr = VecAXPY(X,th->Gamma*dt,th->V1);CHKERRQ(ierr);
  ierr = VecAXPY(X,(1-th->Gamma)*dt,th->V0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSReset_Alpha"
static PetscErrorCode TSReset_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&th->X0);CHKERRQ(ierr);
  ierr = VecDestroy(&th->Xa);CHKERRQ(ierr);
  ierr = VecDestroy(&th->X1);CHKERRQ(ierr);
  ierr = VecDestroy(&th->V0);CHKERRQ(ierr);
  ierr = VecDestroy(&th->Va);CHKERRQ(ierr);
  ierr = VecDestroy(&th->V1);CHKERRQ(ierr);
  ierr = VecDestroy(&th->vec_sol_prev);CHKERRQ(ierr);
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

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlphaUseAdapt_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlphaSetRadius_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlphaSetParams_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlphaGetParams_C",NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormFunction_Alpha"
static PetscErrorCode SNESTSFormFunction_Alpha(SNES snes,Vec x,Vec y,TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  Vec            X0  = th->X0, V0 = th->V0;
  Vec            X1  = x, V1 = th->V1, R = y;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* V1 = 1/(Gamma*dT)*(X1-X0) + (1-1/Gamma)*V0 */
  ierr = VecWAXPY(V1,-1,X0,X1);CHKERRQ(ierr);
  ierr = VecAXPBY(V1,1-1/th->Gamma,1/(th->Gamma*ts->time_step),V0);CHKERRQ(ierr);
  /* Xa = X0 + Alpha_f*(X1-X0) */
  ierr = VecWAXPY(th->Xa,-1,X0,X1);CHKERRQ(ierr);
  ierr = VecAYPX(th->Xa,th->Alpha_f,X0);CHKERRQ(ierr);
  /* Va = V0 + Alpha_m*(V1-V0) */
  ierr = VecWAXPY(th->Va,-1,V0,V1);CHKERRQ(ierr);
  ierr = VecAYPX(th->Va,th->Alpha_m,V0);CHKERRQ(ierr);
  /* F = Function(ta,Xa,Va) */
  ierr = TSComputeIFunction(ts,th->stage_time,th->Xa,th->Va,R,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecScale(R,1/th->Alpha_f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormJacobian_Alpha"
static PetscErrorCode SNESTSFormJacobian_Alpha(SNES snes,Vec x,
#if PETSC_VERSION_LT(3,5,0)
                                               Mat *A,Mat *B,MatStructure *m,
#else
                                               Mat A,Mat B,
#endif
                                               TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* A,B = Jacobian(ta,Xa,Va) */
#if PETSC_VERSION_LT(3,5,0)
  *m = SAME_NONZERO_PATTERN;
  ierr = TSComputeIJacobian(ts,th->stage_time,th->Xa,th->Va,th->shift,A,B,m,PETSC_FALSE);CHKERRQ(ierr);
#else
  ierr = TSComputeIJacobian(ts,th->stage_time,th->Xa,th->Va,th->shift,A,B,PETSC_FALSE);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_Alpha"
static PetscErrorCode TSSetUp_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&th->X0);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Xa);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->X1);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->V0);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->Va);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->V1);CHKERRQ(ierr);
  if (!th->adapt) {
    ierr = TSAdaptDestroy(&ts->adapt);CHKERRQ(ierr);
    ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
    ierr = TSAdaptSetType(ts->adapt,TSADAPTNONE);CHKERRQ(ierr);
  } else {
    ierr = VecDuplicate(ts->vec_sol,&th->vec_sol_prev);CHKERRQ(ierr);
  }
  ierr = TSGetSNES(ts,&ts->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_Alpha"
static PetscErrorCode TSSetFromOptions_Alpha(TS ts)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Alpha ODE solver options");CHKERRQ(ierr);
  {
    PetscBool flag;
    PetscReal radius = 1.0;
    ierr = PetscOptionsReal("-ts_alpha_radius","spectral radius","TSAlphaSetRadius",radius,&radius,&flag);CHKERRQ(ierr);
    if (flag) { ierr = TSAlphaSetRadius(ts,radius);CHKERRQ(ierr); }
    ierr = PetscOptionsReal("-ts_alpha_alpha_m","Algoritmic parameter alpha_m","TSAlphaSetParams",th->Alpha_m,&th->Alpha_m,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_alpha_f","Algoritmic parameter alpha_f","TSAlphaSetParams",th->Alpha_f,&th->Alpha_f,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ts_alpha_gamma","Algoritmic parameter gamma","TSAlphaSetParams",th->Gamma,&th->Gamma,NULL);CHKERRQ(ierr);
    ierr = TSAlphaSetParams(ts,th->Alpha_m,th->Alpha_f,th->Gamma);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ts_alpha_adapt","Use time-step adaptivity with the Alpha method","",th->adapt,&th->adapt,NULL);CHKERRQ(ierr);
    ierr = TSGetSNES(ts,&ts->snes);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(ts->snes);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSView_Alpha"
static PetscErrorCode TSView_Alpha(TS ts,PetscViewer viewer)
{
  TS_Alpha       *th = (TS_Alpha*)ts->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii)   {ierr = PetscViewerASCIIPrintf(viewer,"  Alpha_m=%g, Alpha_f=%g, Gamma=%g\n",(double)th->Alpha_m,(double)th->Alpha_f,(double)th->Gamma);CHKERRQ(ierr);}
  if (ts->snes) {ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSAlphaUseAdapt_Alpha"
static PetscErrorCode TSAlphaUseAdapt_Alpha(TS ts,PetscBool use)
{
  TS_Alpha *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  if (use == th->adapt) PetscFunctionReturn(0);
  if (ts->setupcalled) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ORDER,"Cannot change adaptivity after TSSetUp()");
  th->adapt = use;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlphaSetRadius_Alpha"
static PetscErrorCode TSAlphaSetRadius_Alpha(TS ts,PetscReal radius)
{
  PetscReal      alpha_m,alpha_f,gamma;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (radius < 0 || radius > 1) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Radius %g not in range [0,1]",(double)radius);
  alpha_m = 0.5*(3-radius)/(1+radius);
  alpha_f = 1/(1+radius);
  gamma   = 0.5 + alpha_m - alpha_f;
  ierr = TSAlphaSetParams(ts,alpha_m,alpha_f,gamma);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlphaSetParams_Alpha"
static PetscErrorCode TSAlphaSetParams_Alpha(TS ts,PetscReal alpha_m,PetscReal alpha_f,PetscReal gamma)
{
  TS_Alpha *th = (TS_Alpha*)ts->data;
  PetscReal tol = 100*PETSC_MACHINE_EPSILON;
  PetscReal res = gamma - (0.5 + alpha_m - alpha_f);

  PetscFunctionBegin;
  th->Alpha_m = alpha_m;
  th->Alpha_f = alpha_f;
  th->Gamma   = gamma;
  th->order   = (PetscAbsReal(res) < tol) ? 2 : 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlphaGetParams_Alpha"
static PetscErrorCode TSAlphaGetParams_Alpha(TS ts,PetscReal *alpha_m,PetscReal *alpha_f,PetscReal *gamma)
{
  TS_Alpha *th = (TS_Alpha*)ts->data;

  PetscFunctionBegin;
  if (alpha_m) *alpha_m = th->Alpha_m;
  if (alpha_f) *alpha_f = th->Alpha_f;
  if (gamma)   *gamma   = th->Gamma;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      TSALPHA - DAE solver using the implicit Generalized-Alpha
                method for first-order systems

  Level: beginner

  References:
  K.E. Jansen, C.H. Whiting, G.M. Hulber, "A generalized-alpha
  method for integrating the filtered Navier-Stokes equations with a
  stabilized finite element method", Computer Methods in Applied
  Mechanics and Engineering, 190, 305-319, 2000.
  DOI: 10.1016/S0045-7825(00)00203-6.

  J. Chung, G.M.Hubert. "A Time Integration Algorithm for Structural
  Dynamics with Improved Numerical Dissipation: The Generalized-alpha
  Method" ASME Journal of Applied Mechanics, 60, 371:375, 1993.

.seealso:  TSCreate(), TS, TSSetType()

M*/
#undef __FUNCT__
#define __FUNCT__ "TSCreate_Alpha1"
PETSC_EXTERN PetscErrorCode TSCreate_Alpha1(TS ts)
{
  TS_Alpha       *th;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_Alpha;
  ts->ops->destroy        = TSDestroy_Alpha;
  ts->ops->view           = TSView_Alpha;
  ts->ops->setup          = TSSetUp_Alpha;
  ts->ops->step           = TSStep_Alpha;
  ts->ops->evaluatestep   = TSEvaluateStep_Alpha;
#if 0==PETSC_VERSION_LT(3,5,0)
  ts->ops->rollback       = TSRollBack_Alpha;
#endif
  ts->ops->interpolate    = TSInterpolate_Alpha;
  ts->ops->setfromoptions = TSSetFromOptions_Alpha;
  ts->ops->snesfunction   = SNESTSFormFunction_Alpha;
  ts->ops->snesjacobian   = SNESTSFormJacobian_Alpha;

#if PETSC_VERSION_LT(3,5,0)
  ierr = PetscNewLog(ts,TS_Alpha,&th);CHKERRQ(ierr);
#else
  ierr = PetscNewLog(ts,&th);CHKERRQ(ierr);
#endif
  ts->data = (void*)th;

  th->Alpha_m = 0.5;
  th->Alpha_f = 0.5;
  th->Gamma   = 0.5;

  th->adapt = PETSC_FALSE;
  th->order = 2;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlphaUseAdapt_C",TSAlphaUseAdapt_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlphaSetRadius_C",TSAlphaSetRadius_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlphaSetParams_C",TSAlphaSetParams_Alpha);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSAlphaGetParams_C",TSAlphaGetParams_Alpha);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlphaUseAdapt"
/*@
  TSAlphaUseAdapt - Use time-step adaptivity with the Alpha method

  Logically Collective on TS

  Input Parameter:
+  ts - timestepping context
-  use - flag to use adaptivity

  Options Database:
.  -ts_alpha_adapt

  Level: intermediate

.seealso: TSAdapt
@*/
PetscErrorCode TSAlphaUseAdapt(TS ts,PetscBool use)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveBool(ts,use,2);
  ierr = PetscTryMethod(ts,"TSAlphaUseAdapt_C",(TS,PetscBool),(ts,use));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0 /* XXX */

#undef __FUNCT__
#define __FUNCT__ "TSAlphaSetRadius"
/*@
  TSAlphaSetRadius - sets the desired spectral radius of the method
                     (i.e. high-frequency numerical damping)

  Logically Collective on TS

  The algorithmic parameters \alpha_m and \alpha_f of the
  generalized-\alpha method can be computed in terms of a specified
  spectral radius \rho in [0,1] for infinite time step in order to
  control high-frequency numerical damping:
    alpha_m = 0.5*(3-\rho)/(1+\rho)
    alpha_f = 1/(1+\rho)

  Input Parameter:
+  ts - timestepping context
-  radius - the desired spectral radius

  Options Database:
.  -ts_alpha_radius <radius>

  Level: intermediate

.seealso: TSAlphaSetParams(), TSAlphaGetParams()
@*/
PetscErrorCode TSAlphaSetRadius(TS ts,PetscReal radius)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,radius,2);
  ierr = PetscTryMethod(ts,"TSAlphaSetRadius_C",(TS,PetscReal),(ts,radius));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlphaSetParams"
/*@
  TSAlphaSetParams - sets the algorithmic parameters for TSALPHA

  Logically Collective on TS

  Second-order accuracy can be obtained so long as:
    \gamma = 0.5 + alpha_m - alpha_f

  Unconditional stability requires:
    \alpha_m >= \alpha_f >= 0.5

  Backward Euler method is recovered when:
    \alpha_m = \alpha_f = gamma = 1


  Input Parameter:
+  ts - timestepping context
.  \alpha_m - algorithmic paramenter
.  \alpha_f - algorithmic paramenter
-  \gamma   - algorithmic paramenter

   Options Database:
+  -ts_alpha_alpha_m <alpha_m>
.  -ts_alpha_alpha_f <alpha_f>
-  -ts_alpha_gamma <gamma>

  Note:
  Use of this function is normally only required to hack TSALPHA to
  use a modified integration scheme. Users should call
  TSAlphaSetRadius() to set the desired spectral radius of the methods
  (i.e. high-frequency damping) in order so select optimal values for
  these parameters.

  Level: advanced

.seealso: TSAlphaSetRadius(), TSAlphaGetParams()
@*/
PetscErrorCode TSAlphaSetParams(TS ts,PetscReal alpha_m,PetscReal alpha_f,PetscReal gamma)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,alpha_m,2);
  PetscValidLogicalCollectiveReal(ts,alpha_f,3);
  PetscValidLogicalCollectiveReal(ts,gamma,4);
  ierr = PetscTryMethod(ts,"TSAlphaSetParams_C",(TS,PetscReal,PetscReal,PetscReal),(ts,alpha_m,alpha_f,gamma));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAlphaGetParams"
/*@
  TSAlphaGetParams - gets the algorithmic parameters for TSALPHA

  Not Collective

  Input Parameter:
+  ts - timestepping context
.  \alpha_m - algorithmic parameter
.  \alpha_f - algorithmic parameter
-  \gamma   - algorithmic parameter

  Note:
  Use of this function is normally only required to hack TSALPHA to
  use a modified integration scheme. Users should call
  TSAlphaSetRadius() to set the high-frequency damping (i.e. spectral
  radius of the method) in order so select optimal values for these
  parameters.

  Level: advanced

.seealso: TSAlphaSetRadius(), TSAlphaSetParams()
@*/
PetscErrorCode TSAlphaGetParams(TS ts,PetscReal *alpha_m,PetscReal *alpha_f,PetscReal *gamma)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (alpha_m) PetscValidPointer(alpha_m,2);
  if (alpha_f) PetscValidPointer(alpha_f,3);
  if (gamma)   PetscValidPointer(gamma,4);
  ierr = PetscUseMethod(ts,"TSAlphaGetParams_C",(TS,PetscReal*,PetscReal*,PetscReal*),(ts,alpha_m,alpha_f,gamma));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* XXX */
