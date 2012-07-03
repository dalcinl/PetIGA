#if !defined(PETIGAPART_H)
#define PETIGAPART_H

#include <petsc.h>

#if PETSC_VERSION_(3,2,0)
PETSC_EXTERN_CXX_BEGIN
#endif

#ifndef PETSC_EXTERN
#define PETSC_EXTERN extern
#endif

PETSC_EXTERN PetscErrorCode IGA_Partition(PetscInt,PetscInt,
                                          PetscInt,const PetscInt[],
                                          PetscInt[],PetscInt[]);
PETSC_EXTERN PetscErrorCode IGA_Distribute(PetscInt,
                                           const PetscInt[],const PetscInt[],
                                           const PetscInt[],PetscInt[],PetscInt[]);

#if PETSC_VERSION_(3,2,0)
PETSC_EXTERN_CXX_BEGIN
#endif

#endif/*PETIGAPART_H*/
