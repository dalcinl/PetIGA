#if !defined(__PETSCTS1_H)
#define __PETSCTS1_H
#include <petscts.h>

#define TSALPHA1 "alpha1"
#if PETSC_VERSION_LT(3,7,0)
PETSC_EXTERN PetscErrorCode TSAlphaUseAdapt(TS,PetscBool);
#endif

#define TSBDF "bdf"
PETSC_EXTERN PetscErrorCode TSBDFSetOrder(TS,PetscInt);
PETSC_EXTERN PetscErrorCode TSBDFGetOrder(TS,PetscInt*);
PETSC_EXTERN PetscErrorCode TSBDFUseAdapt(TS,PetscBool);

#endif/*__PETSCTS1_H*/
