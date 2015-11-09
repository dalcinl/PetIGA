/*
  Code for timestepping with BDF methods
*/
#include <petscts1.h>
#if PETSC_VERSION_LT(3,6,0)
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/
#else
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#endif

/*
static PetscBool  cited = PETSC_FALSE;
static const char citation[] = "";
*/

#if PETSC_VERSION_LT(3,5,0)
#define PetscCitationsRegister(a,b) ((void)a,(void)b,0)
#define TSPostStage(ts,t,n,x) 0
static PetscErrorCode TSRollBack_BDF(TS);
#define TSRollBack(ts) \
  TSRollBack_BDF(ts); \
  ts->ptime -= next_time_step; \
  ts->time_step = next_time_step;
#endif

#if PETSC_VERSION_LT(3,7,0)
#define TSAdaptCheckStage(adapt,ts,t,X,accept) TSAdaptCheckStage(adapt,ts,accept)
#endif

typedef struct {
  PetscInt  order;
  PetscReal time[4+1];
  Vec       work[4+1];
  PetscReal shift;
  Vec       vec_dot;
  Vec       vec_sol;

  PetscBool    adapt;
  TSStepStatus status;
} TS_BDF;

#undef __FUNCT__
#define __FUNCT__ "TSBDF_VecDot"
static PetscErrorCode TSBDF_VecDot(TS ts,PetscInt order,PetscReal t,Vec X,Vec Xdot,PetscReal *shift)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  PetscInt       i,n = order+1;
  PetscReal      *time = th->time;
  Vec            vecs[6];
  PetscReal      time_step = t - time[1];
  PetscReal      a = (time[1]-time[2])/time_step;
  PetscReal      b = (time[2]-time[3])/time_step;
  PetscScalar    alpha[6];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (order) {
  case 1:
    alpha[0] = +1.0;
    alpha[1] = -1.0;
    break;
  case 2:
    alpha[0] = +(a+2)/(a+1) ;
    alpha[1] = -(a+1)/a     ;
    alpha[2] = +1/(a*(a+1)) ;
    break;
  case 3:
    alpha[0] = +1+1/(a+1)+1/(a+b+1)     ;
    alpha[1] = -(a+1)*(a+b+1)/(a*(a+b)) ;
    alpha[2] = +(a+b+1)/(a*b*(a+1))     ;
    alpha[3] = -(a+1)/(b*(a+b)*(a+b+1)) ;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"BDF Order %D not implemented",order);
  }

  vecs[0] = X;
  for (i=1; i<n; i++) vecs[i]   = th->work[i];
  for (i=0; i<n; i++) alpha[i] /= time_step;

  if (shift) *shift = PetscRealPart(alpha[0]);
  ierr = VecZeroEntries(Xdot);CHKERRQ(ierr);
  ierr = VecMAXPY(Xdot,n,alpha,vecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSBDF_VecLTE"
static PetscErrorCode TSBDF_VecLTE(TS ts,PetscInt order,Vec lte)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  PetscInt       i,n = order+1;
  PetscReal      *time = th->time;
  Vec            *vecs = th->work;
  PetscReal      h = (time[0]-time[1]);
  PetscReal      a = (time[1]-time[2])/h;
  PetscReal      b = (time[2]-time[3])/h;
  PetscReal      c = (time[3]-time[4])/h;
  PetscReal      scale;
  PetscScalar    alpha[6];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (order) {
  case 1:
    scale    = -1.0 ;
    alpha[0] = +1.0 ;
    alpha[1] = -1.0 ;
    break;
  case 2:
    scale    = -1.0/2       ;
    alpha[0] = +2/(a+1)     ;
    alpha[1] = -2/a         ;
    alpha[2] = +2/(a*(a+1)) ;
    break;
  case 3:
    scale    = -(a+1)*(a+1)/(6*(a+2)) ;
    alpha[0] = +6/((a+1)*(a+b+1))     ;
    alpha[1] = -6/(a*(a+b))           ;
    alpha[2] = +6/(a*b*(a+1))         ;
    alpha[3] = -6/(b*(a+b)*(a+b+1))   ;
    break;
  case 4:
    scale    = -(a+1)*(a+1)*(a+b+1)*(a+b+1)/(24*(a*a+a*b+4*a+2*b+3)) ;
    alpha[0] = +24/((a+1)*(a+b+1)*(a+b+c+1))                         ;
    alpha[1] = -24/(a*(a+b)*(a+b+c))                                 ;
    alpha[2] = +24/(a*b*(a+1)*(b+c))                                 ;
    alpha[3] = -24/(b*c*(a+b)*(a+b+1))                               ;
    alpha[4] = +24/(c*(b+c)*(a+b+c)*(a+b+c+1))                       ;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"BDF Order %D not implemented ",order);
  }
  for (i=0; i<n; i++) alpha[i] *= scale;
  ierr = VecZeroEntries(lte);CHKERRQ(ierr);
  ierr = VecMAXPY(lte,n,alpha,vecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSBDF_Predictor"
static PetscErrorCode TSBDF_Predictor(TS ts,PetscInt order,PetscReal t,Vec X)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  PetscInt       n = order;
  PetscReal      *time = th->time+1;
  Vec            *vecs = th->work+1;
  PetscReal      h = (time[0]-time[1]);
  PetscReal      a = (time[1]-time[2])/h;
  PetscReal      b = (time[2]-time[3])/h;
  PetscReal      s = (t-time[0])/h;
  PetscScalar    alpha[6];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (order) {
  case 1:
    ierr = VecCopy(vecs[0],X);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  case 2:
    alpha[0] = +s+1 ;
    alpha[1] = -s   ;
    break;
  case 3:
    alpha[0] = +(s+1)*(a+s+1)/(a+1) ;
    alpha[1] = -s*(a+s+1)/a         ;
    alpha[2] = +s*(s+1)/(a*(a+1))   ;
    break;
  case 4:
    alpha[0] = +(s+1)*(a+s+1)*(a+b+s+1)/((a+1)*(a+b+1)) ;
    alpha[1] = -s*(a+s+1)*(a+b+s+1)/(a*(a+b))           ;
    alpha[2] = +s*(s+1)*(a+b+s+1)/(a*b*(a+1))           ;
    alpha[3] = -s*(s+1)*(a+s+1)/(b*(a+b)*(a+b+1))       ;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"BDF Order %D not implemented ",order);
  }
  ierr = VecZeroEntries(X);CHKERRQ(ierr);
  ierr = VecMAXPY(X,n,alpha,vecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSBDF_Interpolate"
static PetscErrorCode TSBDF_Interpolate(TS ts,PetscInt order,PetscReal t,Vec X)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  PetscInt       n = order+1;
  PetscReal      *time = th->time;
  Vec            *vecs = th->work;
  PetscReal      h = (time[0]-time[1]);
  PetscReal      a = (time[1]-time[2])/h;
  PetscReal      b = (time[2]-time[3])/h;
  PetscReal      s = (t-time[1])/h;
  PetscScalar    alpha[6];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (order) {
  case 1:
    alpha[0] = +s   ;
    alpha[1] = -s+1 ;
    break;
  case 2:
    alpha[0] = +s*(a+s)/(a+1)     ;
    alpha[1] = -(a+s)*(s-1)/a     ;
    alpha[2] = +s*(s-1)/(a*(a+1)) ;
    break;
  case 3:
    alpha[0] = +s*(a+s)*(a+b+s)/((a+1)*(a+b+1)) ;
    alpha[1] = -(a+s)*(s-1)*(a+b+s)/(a*(a+b))   ;
    alpha[2] = +s*(s-1)*(a+b+s)/(a*b*(a+1))     ;
    alpha[3] = -s*(a+s)*(s-1)/(b*(a+b)*(a+b+1)) ;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"BDF Order %D not implemented ",order);
  }
  ierr = VecZeroEntries(X);CHKERRQ(ierr);
  ierr = VecMAXPY(X,n,alpha,vecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BDF_ClipTimestep"
static PetscReal BDF_ClipTimestep(PetscInt order,PetscReal h,PetscReal next_h)
{
  const PetscInt  k = order-2;
  const PetscReal R[] = {
    /* k = 2 */ 2.414, /*  1+sqrt(2)    */
    /* k = 3 */ 1.618, /* (1+sqrt(5))/2 */
    /* k = 4 */ 1.280, /*               */
    1.127, 1.044 };
  return (k >= 0 && next_h/h > R[k]) ? h*R[k] : next_h;
}

#undef __FUNCT__
#define __FUNCT__ "TSBDF_Advance"
static PetscErrorCode TSBDF_Advance(TS ts,PetscReal t,Vec X)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  size_t         i,n = sizeof(th->work)/sizeof(Vec);
  Vec            tail = th->work[n-1];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (X) {ierr = VecCopy(X,tail);CHKERRQ(ierr);}
  for (i=n-1; i>=2; i--) {
    th->time[i] = th->time[i-1];
    th->work[i] = th->work[i-1];
  }
  th->time[1] = t;
  th->work[1] = tail;
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
#define __FUNCT__ "TSBDF_SolveStep"
static PetscErrorCode TSBDF_SolveStep(TS ts,PetscReal t,Vec X,PetscBool *stageok)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSPreStage(ts,t);CHKERRQ(ierr);
  ierr = TS_SNESSolve(ts,NULL,X);CHKERRQ(ierr);
  ierr = TSPostStage(ts,t,0,&X);CHKERRQ(ierr);
  ierr = TSAdaptCheckStage(ts->adapt,ts,t,X,stageok);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSBDF_AdaptStep"
static PetscErrorCode TSBDF_AdaptStep(TS ts,PetscInt order,PetscReal *next_h,PetscBool *accept)
{
  PetscInt       next_scheme;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSAdaptCandidatesClear(ts->adapt);CHKERRQ(ierr);
  ierr = TSAdaptCandidateAdd(ts->adapt,"",order,1,1.0,1.0,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSAdaptChoose(ts->adapt,ts,ts->time_step,&next_scheme,next_h,accept);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSBDF_InitialStep"
static PetscErrorCode TSBDF_InitialStep(TS ts,PetscBool *stageok)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  PetscInt       i,order = th->order;
  PetscReal      tau,ratio = 1 + PetscPowReal(2,2-order); /*3,2,1.5,1.25*/
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = TSBDF_Advance(ts,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
  for (tau=0, i=0; i<=order; i++) tau = 1 + tau*ratio;
  ts->time_step /= tau;

  th->order = 0;
  while (++th->order <= order) {
    th->time[0] = ts->ptime + ts->time_step;
    ierr = TSBDF_Predictor(ts,th->order,th->time[0],th->work[0]);CHKERRQ(ierr);
    ierr = TSBDF_SolveStep(ts,th->time[0],th->work[0],stageok);CHKERRQ(ierr);
    if (!*stageok) break;
    ierr = TSBDF_Advance(ts,th->time[0],th->work[0]);CHKERRQ(ierr);
    ts->ptime += ts->time_step;
    ts->time_step *= ratio;
  }
  th->order = order;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSStep_BDF"
static PetscErrorCode TSStep_BDF(TS ts)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  PetscBool      stageok,accept = PETSC_TRUE;
  PetscInt       rejections = 0;
  PetscInt       order = th->order;
  PetscReal      next_time_step = ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);*/

  th->status = TS_STEP_INCOMPLETE;
  ierr = VecCopy(ts->vec_sol,th->vec_sol);CHKERRQ(ierr);
  ierr = TSBDF_Advance(ts,ts->ptime,ts->vec_sol);CHKERRQ(ierr);

  if (!th->adapt) th->order = PetscMin(th->order,ts->steps+1);
  while (!ts->reason && th->status != TS_STEP_COMPLETE) {
    ierr = TSPreStep(ts);CHKERRQ(ierr);

    if (th->adapt && ts->steps == 0) {
      ierr = TSBDF_InitialStep(ts,&stageok);CHKERRQ(ierr);
      if (!stageok) {accept = PETSC_FALSE; goto reject_step;}
    }

    th->time[0] = ts->ptime + ts->time_step;
    ierr = TSBDF_Predictor(ts,th->order,th->time[0],th->work[0]);CHKERRQ(ierr);
    ierr = TSBDF_SolveStep(ts,th->time[0],th->work[0],&stageok);CHKERRQ(ierr);
    if (!stageok) {accept = PETSC_FALSE; goto reject_step;}

    th->status = TS_STEP_PENDING;
    ierr = VecCopy(th->work[0],ts->vec_sol);CHKERRQ(ierr);
    ierr = TSBDF_AdaptStep(ts,th->order,&next_time_step,&accept);CHKERRQ(ierr);
    if (!accept) {
      th->status = TS_STEP_INCOMPLETE;
      ts->ptime += next_time_step;
      ierr = TSRollBack(ts);CHKERRQ(ierr);
      ts->time_step = next_time_step;
      goto reject_step;
    }

    th->status = TS_STEP_COMPLETE;
    th->order = order;
    ts->ptime += ts->time_step;
    ts->time_step = BDF_ClipTimestep(th->order,ts->time_step,next_time_step);
    ts->steps++;
    break;

  reject_step:
    ts->reject++;
    if (!ts->reason && ++rejections > ts->max_reject && ts->max_reject >= 0) {
      ierr = PetscInfo2(ts,"Step=%D, step rejections %D greater than current TS allowed, stopping solve\n",ts->steps,rejections);CHKERRQ(ierr);
      ts->reason = TS_DIVERGED_STEP_REJECTED;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSEvaluateStep_BDF"
static PetscErrorCode TSEvaluateStep_BDF(TS ts,PetscInt order,Vec X,PETSC_UNUSED PetscBool *done)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSBDF_VecLTE(ts,order+1,X);CHKERRQ(ierr);
  ierr = VecAXPY(X,1.0,th->work[0]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSInterpolate_BDF"
static PetscErrorCode TSInterpolate_BDF(TS ts,PetscReal t,Vec X)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  PetscInt       order = PetscMin(th->order,ts->steps);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSBDF_Interpolate(ts,order,t,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRollBack_BDF"
static PetscErrorCode TSRollBack_BDF(TS ts)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(th->vec_sol,ts->vec_sol);CHKERRQ(ierr);
  th->status = TS_STEP_INCOMPLETE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormFunction_BDF"
static PetscErrorCode SNESTSFormFunction_BDF(PETSC_UNUSED SNES snes,Vec X,Vec F,TS ts)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  PetscInt       order = th->order;
  PetscReal      t = th->time[0];
  Vec            V = th->vec_dot;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSBDF_VecDot(ts,order,t,X,V,&th->shift);CHKERRQ(ierr);
  /* F = Function(t,X,V) */
  ierr = TSComputeIFunction(ts,t,X,V,F,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormJacobian_BDF"
static PetscErrorCode SNESTSFormJacobian_BDF(PETSC_UNUSED SNES snes,
                                             PETSC_UNUSED Vec X,
#if PETSC_VERSION_LT(3,5,0)
                                             Mat *J,Mat *P,MatStructure *m,
#else
                                             Mat J,Mat P,
#endif
                                             TS ts)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  PetscReal      t = th->time[0];
  Vec            V = th->vec_dot;
  PetscReal      dVdX = th->shift;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* J,P = Jacobian(t,X,V) */
#if PETSC_VERSION_LT(3,5,0)
  *m = SAME_NONZERO_PATTERN;
  ierr = TSComputeIJacobian(ts,t,X,V,dVdX,J,P,m,PETSC_FALSE);CHKERRQ(ierr);
#else
  ierr = TSComputeIJacobian(ts,t,X,V,dVdX,J,P,PETSC_FALSE);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSReset_BDF"
static PetscErrorCode TSReset_BDF(TS ts)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  size_t         i,n = sizeof(th->work)/sizeof(Vec);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {ierr = VecDestroy(&th->work[i]);CHKERRQ(ierr);}
  ierr = VecDestroy(&th->vec_dot);CHKERRQ(ierr);
  ierr = VecDestroy(&th->vec_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_BDF"
static PetscErrorCode TSDestroy_BDF(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_BDF(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSBDFSetOrder_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSBDFGetOrder_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSBDFUseAdapt_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscBool always_accept;
  PetscReal clip[2];
  PetscReal safety;
  PetscReal reject_safety;
  Vec       Y;
} TSAdapt_Basic;

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_BDF"
static PetscErrorCode TSSetUp_BDF(TS ts)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  size_t         i,n = sizeof(th->work)/sizeof(Vec);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {ierr = VecDuplicate(ts->vec_sol,&th->work[i]);CHKERRQ(ierr);}
  ierr = VecDuplicate(ts->vec_sol,&th->vec_dot);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&th->vec_sol);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
  if (!th->adapt) {
    ierr = TSAdaptDestroy(&ts->adapt);CHKERRQ(ierr);
    ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
    ierr = TSAdaptSetType(ts->adapt,TSADAPTNONE);CHKERRQ(ierr);
  } else {
    PetscBool isbasic; TSAdapt_Basic *basic = (TSAdapt_Basic*)ts->adapt->data;
    ierr = PetscObjectTypeCompare((PetscObject)ts->adapt,TSADAPTBASIC,&isbasic);CHKERRQ(ierr);
    if (isbasic)  basic->clip[1] = BDF_ClipTimestep(th->order,1,basic->clip[1]);
  }

  ierr = TSGetSNES(ts,&ts->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if PETSC_VERSION_LT(3,7,0)
typedef PetscOptions PetscOptionItems;
#endif
#if PETSC_VERSION_LT(3,6,0)
#define PetscOptionsHead(obj,head) ((void)(obj),PetscOptionsHead(head))
#endif
#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_BDF"
static PetscErrorCode TSSetFromOptions_BDF(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"BDF ODE solver options");CHKERRQ(ierr);
  {
    PetscBool flg;
    PetscInt  order = th->order;
    PetscBool adapt = th->adapt;
    ierr = PetscOptionsInt("-ts_bdf_order","Order of the BDF method","TSBDFSetOrder",order,&order,&flg);CHKERRQ(ierr);
    if (flg) {ierr = TSBDFSetOrder(ts,order);CHKERRQ(ierr);}
    ierr = PetscOptionsBool("-ts_bdf_adapt","Use time-step adaptivity with the BDF method","TSBDFUseAdapt",adapt,&adapt,&flg);CHKERRQ(ierr);
    if (flg) {ierr = TSBDFUseAdapt(ts,adapt);CHKERRQ(ierr);}
#if PETSC_VERSION_LT(3,6,0)
    ierr = SNESSetFromOptions(ts->snes);CHKERRQ(ierr);
#endif
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#if PETSC_VERSION_LT(3,6,0)
static PetscErrorCode TSSetFromOptions_BDF_Legacy(TS ts) {return TSSetFromOptions_BDF(NULL,ts);}
#define TSSetFromOptions_BDF TSSetFromOptions_BDF_Legacy
#endif

#undef __FUNCT__
#define __FUNCT__ "TSView_BDF"
static PetscErrorCode TSView_BDF(TS ts,PetscViewer viewer)
{
  TS_BDF         *th = (TS_BDF*)ts->data;
  PetscBool      ascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&ascii);CHKERRQ(ierr);
  if (ascii)    {ierr = PetscViewerASCIIPrintf(viewer,"  Order=%D\n",th->order);CHKERRQ(ierr);}
  if (ts->snes) {ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

#undef __FUNCT__
#define __FUNCT__ "TSBDFSetOrder_BDF"
static PetscErrorCode TSBDFSetOrder_BDF(TS ts,PetscInt order)
{
  TS_BDF *th = (TS_BDF*)ts->data;

  PetscFunctionBegin;
  if (order == th->order) PetscFunctionReturn(0);
  if (order < 1 || order > 3) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"BDF Order %D not implemented",order);
  th->order = order;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSBDFGetOrder_BDF"
static PetscErrorCode TSBDFGetOrder_BDF(TS ts,PetscInt *order)
{
  TS_BDF *th = (TS_BDF*)ts->data;

  PetscFunctionBegin;
  *order = th->order;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSBDFUseAdapt_BDF"
static PetscErrorCode TSBDFUseAdapt_BDF(TS ts,PetscBool use)
{
  TS_BDF *th = (TS_BDF*)ts->data;

  PetscFunctionBegin;
  if (use == th->adapt) PetscFunctionReturn(0);
  if (ts->setupcalled) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ORDER,"Cannot change adaptivity after TSSetUp()");
  th->adapt = use;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

/*MC
      TSBDF - DAE solver using BDF formulas

  Level: beginner

  References:

.seealso:  TS, TSCreate(), TSSetType()
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSCreate_BDF"
PetscErrorCode TSCreate_BDF(TS ts);
PetscErrorCode TSCreate_BDF(TS ts)
{
  TS_BDF         *th;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_BDF;
  ts->ops->destroy        = TSDestroy_BDF;
  ts->ops->view           = TSView_BDF;
  ts->ops->setup          = TSSetUp_BDF;
  ts->ops->setfromoptions = TSSetFromOptions_BDF;
  ts->ops->step           = TSStep_BDF;
  ts->ops->evaluatestep   = TSEvaluateStep_BDF;
#if PETSC_VERSION_GE(3,5,0)
  ts->ops->rollback       = TSRollBack_BDF;
#endif
  ts->ops->interpolate    = TSInterpolate_BDF;
  ts->ops->snesfunction   = SNESTSFormFunction_BDF;
  ts->ops->snesjacobian   = SNESTSFormJacobian_BDF;

#if PETSC_VERSION_LT(3,5,0)
  ierr = PetscNewLog(ts,TS_BDF,&th);CHKERRQ(ierr);
#else
  ierr = PetscNewLog(ts,&th);CHKERRQ(ierr);
#endif
  ts->data = (void*)th;

  th->order  = 2;
  th->adapt  = PETSC_FALSE;
  th->status = TS_STEP_COMPLETE;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSBDFSetOrder_C",TSBDFSetOrder_BDF);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSBDFGetOrder_C",TSBDFGetOrder_BDF);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSBDFUseAdapt_C",TSBDFUseAdapt_BDF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ------------------------------------------------------------ */

#undef __FUNCT__
#define __FUNCT__ "TSBDFSetOrder"
/*@
  TSBDFSetOrder - Set the order of the BDF method

  Logically Collective on TS

  Input Parameter:
+  ts - timestepping context
-  order - order of the method

  Options Database:
.  -ts_bdf_order <order>

  Level: intermediate

@*/
PetscErrorCode TSBDFSetOrder(TS ts,PetscInt order)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ts,order,2);
  ierr = PetscTryMethod(ts,"TSBDFSetOrder_C",(TS,PetscInt),(ts,order));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSBDFGetOrder"
/*@
  TSBDFGetOrder - Get the order of the BDF method

  Not Collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  order - order of the method

  Level: intermediate

@*/
PetscErrorCode TSBDFGetOrder(TS ts,PetscInt *order)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(order,2);
  ierr = PetscUseMethod(ts,"TSBDFGetOrder_C",(TS,PetscInt*),(ts,order));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSBDFUseAdapt"
/*@
  TSBDFUseAdapt - Use time-step adaptivity with the BDF method

  Logically Collective on TS

  Input Parameter:
+  ts - timestepping context
-  use - flag to use adaptivity

  Options Database:
.  -ts_bdf_adapt

  Level: intermediate

.seealso: TSAdapt
@*/
PetscErrorCode TSBDFUseAdapt(TS ts,PetscBool use)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveBool(ts,use,2);
  ierr = PetscTryMethod(ts,"TSBDFUseAdapt_C",(TS,PetscBool),(ts,use));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
