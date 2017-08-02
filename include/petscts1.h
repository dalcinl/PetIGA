#if !defined(__PETSCTS1_H)
#define __PETSCTS1_H
#include <petscts.h>

#if PETSC_VERSION_LT(3,8,0)
PETSC_EXTERN PetscErrorCode TSSetMaxSteps(TS,PetscInt);
PETSC_EXTERN PetscErrorCode TSGetMaxSteps(TS,PetscInt*);
PETSC_EXTERN PetscErrorCode TSSetMaxTime(TS,PetscReal);
PETSC_EXTERN PetscErrorCode TSGetMaxTime(TS,PetscReal*);
#endif

#define TSALPHA1 "alpha1"
#if PETSC_VERSION_LT(3,7,0)
PETSC_EXTERN PetscErrorCode TSAlphaUseAdapt(TS,PetscBool);
#endif

#if PETSC_VERSION_LT(3,7,0)
#define TSBDF "bdf"
PETSC_EXTERN PetscErrorCode TSBDFSetOrder(TS,PetscInt);
PETSC_EXTERN PetscErrorCode TSBDFGetOrder(TS,PetscInt*);
PETSC_EXTERN PetscErrorCode TSBDFUseAdapt(TS,PetscBool);
#endif

#endif/*__PETSCTS1_H*/
