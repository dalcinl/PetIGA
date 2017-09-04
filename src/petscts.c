#include "petiga.h"
#if PETSC_VERSION_LT(3,8,0)

PetscErrorCode TSSetMaxSteps(TS ts,PetscInt maxsteps)  {return TSSetDuration(ts,maxsteps,PETSC_DEFAULT);}
PetscErrorCode TSGetMaxSteps(TS ts,PetscInt *maxsteps) {return TSGetDuration(ts,maxsteps,NULL);}
PetscErrorCode TSSetMaxTime(TS ts,PetscReal maxtime)   {return TSSetDuration(ts,PETSC_DEFAULT,maxtime);}
PetscErrorCode TSGetMaxTime(TS ts,PetscReal *maxtime)  {return TSGetDuration(ts,NULL,maxtime);}

#else

PetscErrorCode TSAlphaUseAdapt(TS ts,PetscBool flag)
{
  TSAdapt adapt;
  PetscBool match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)ts,TSALPHA,&match);CHKERRQ(ierr);
  if (!match) PetscFunctionReturn(0);
  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,flag ? TSADAPTBASIC : TSADAPTNONE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif
