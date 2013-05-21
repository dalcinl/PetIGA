#if !defined(PETIGAPART_H)
#define PETIGAPART_H

#include <petsc.h>

PETSC_EXTERN PetscErrorCode IGA_Partition(PetscInt,PetscInt,
                                          PetscInt,const PetscInt[],
                                          PetscInt[],PetscInt[]);
PETSC_EXTERN PetscErrorCode IGA_Distribute(PetscInt,
                                           const PetscInt[],const PetscInt[],
                                           const PetscInt[],PetscInt[],PetscInt[]);

#endif/*PETIGAPART_H*/
