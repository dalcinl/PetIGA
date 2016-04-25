#if !defined(__PETSCTS2_H)
#define __PETSCTS2_H
#include <petscdm.h>
#include <petscts.h>

#if PETSC_VERSION_LT(3,7,0)
typedef PetscErrorCode (*TSI2Function)(TS,PetscReal,Vec,Vec,Vec,Vec,void*);
typedef PetscErrorCode (*TSI2Jacobian)(TS,PetscReal,Vec,Vec,Vec,PetscReal,PetscReal,Mat,Mat,void*);
PETSC_EXTERN PetscErrorCode DMTSSetI2Function(DM,TSI2Function,void*);
PETSC_EXTERN PetscErrorCode DMTSSetI2Jacobian(DM,TSI2Jacobian,void*);
PETSC_EXTERN PetscErrorCode DMTSGetI2Function(DM,TSI2Function*,void**);
PETSC_EXTERN PetscErrorCode DMTSGetI2Jacobian(DM,TSI2Jacobian*,void**);
PETSC_EXTERN PetscErrorCode TSSetI2Function(TS,Vec,TSI2Function,void*);
PETSC_EXTERN PetscErrorCode TSSetI2Jacobian(TS,Mat,Mat,TSI2Jacobian,void*);
PETSC_EXTERN PetscErrorCode TSComputeI2Function(TS,PetscReal,Vec,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode TSComputeI2Jacobian(TS,PetscReal,Vec,Vec,Vec,PetscReal,PetscReal,Mat,Mat);
PETSC_EXTERN PetscErrorCode TS2SetSolution(TS,Vec,Vec);
PETSC_EXTERN PetscErrorCode TS2GetSolution(TS,Vec*,Vec*);
#endif

#if PETSC_VERSION_LT(3,7,0)
#define TSALPHA2 "alpha2"
PETSC_EXTERN PetscErrorCode TSAlpha2UseAdapt(TS,PetscBool);
PETSC_EXTERN PetscErrorCode TSAlpha2SetRadius(TS,PetscReal);
PETSC_EXTERN PetscErrorCode TSAlpha2SetParams(TS,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode TSAlpha2GetParams(TS,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
#endif

#endif/*__PETSCTS2_H*/
