#include <petscts.h>
#if PETSC_VERSION_LT(3,8,0)

#include <petscts1.h>
#if PETSC_VERSION_LT(3,6,0)
#include <petsc-private/tsimpl.h> /*I   "petscts.h"   I*/
#else
#include <petsc/private/tsimpl.h> /*I   "petscts.h"   I*/
#endif

#undef  __FUNCT__
#define __FUNCT__ PETSC_FUNCTION_NAME

PetscErrorCode TSSetMaxSteps(TS ts,PetscInt maxsteps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ts,maxsteps,2);
  if (maxsteps < 0 ) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Maximum number of steps must be non-negative");
  ts->max_steps = maxsteps;
  PetscFunctionReturn(0);
}

PetscErrorCode TSGetMaxSteps(TS ts,PetscInt *maxsteps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(maxsteps,2);
  *maxsteps = ts->max_steps;
  PetscFunctionReturn(0);
}

PetscErrorCode TSSetMaxTime(TS ts,PetscReal maxtime)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,maxtime,2);
  ts->max_time = maxtime;
  PetscFunctionReturn(0);
}

PetscErrorCode TSGetMaxTime(TS ts,PetscReal *maxtime)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidRealPointer(maxtime,2);
  *maxtime = ts->max_time;
  PetscFunctionReturn(0);
}

#endif/* PETSc >= 3.8 */
