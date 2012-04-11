! -*- f90 -*-

#include "petscconf.h"

! PetscInt
#if defined(PETSC_USE_64BIT_INDICES)
#define C_PETSC_INT C_LONG_LONG
#else
#define C_PETSC_INT C_INT
#endif

! PetscReal
#if defined(PETSC_USE_REAL_SINGLE)
#define C_PETSC_REAL    C_FLOAT
#define C_PETSC_COMPLEX C_FLOAT_COMPLEX
#elif defined(PETSC_USE_REAL_DOUBLE)
#define C_PETSC_REAL    C_DOUBLE
#define C_PETSC_COMPLEX C_DOUBLE_COMPLEX
#elif defined(PETSC_USE_REAL_LONG_DOUBLE)
#define C_PETSC_REAL    C_LONG_DOUBLE
#define C_PETSC_COMPLEX C_LONG_DOUBLE_COMPLEX
#elif defined(PETSC_USE_REAL___FLOAT128)
#define C_PETSC_REAL    C_FLOAT128
#define C_PETSC_COMPLEX C_FLOAT128_COMPLEX
#endif

! PetscScalar
#if defined(PETSC_USE_COMPLEX)
#define C_PETSC_SCALAR  C_PETSC_COMPLEX
#else
#define C_PETSC_SCALAR  C_PETSC_REAL
#endif

module PetIGA
  use ISO_C_BINDING, only: IGA_INT     => C_PETSC_INT
  use ISO_C_BINDING, only: IGA_REAL    => C_PETSC_REAL
  use ISO_C_BINDING, only: IGA_SCALAR  => C_PETSC_SCALAR
  use ISO_C_BINDING, only: IGA_COMPLEX => C_PETSC_COMPLEX
  implicit none
end module PetIGA
