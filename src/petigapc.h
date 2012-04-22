#include "petscblaslapack.h"
#if defined(PETSC_BLASLAPACK_UNDERSCORE)
   #define sgetri_ sgetri_
   #define dgetri_ dgetri_
   #define qgetri_ qgetri_
   #define cgetri_ cgetri_
   #define zgetri_ zgetri_
#elif defined(PETSC_BLASLAPACK_CAPS)
   #define sgetri_ SGETRI
   #define dgetri_ DGETRI
   #define qgetri_ QGETRI
   #define cgetri_ CGETRI
   #define zgetri_ ZGETRI
#else /* (PETSC_BLASLAPACK_C) */
   #define sgetri_ sgetri
   #define dgetri_ dgetri
   #define qgetri_ qgetri
   #define cgetri_ cgetri
   #define zgetri_ zgetri
#endif
#if !defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define LAPACKgetri_ sgetri_
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define LAPACKgetri_ dgetri_
  #else /* (PETSC_USE_REAL_QUAD) */
    #define LAPACKgetri_ qgetri_
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
    #define LAPACKgetri_ cgetri_
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define LAPACKgetri_ zgetri_
  #else /* (PETSC_USE_REAL_QUAD) */
    #error "LAPACKgetri_ not defined for quad complex"
  #endif
#endif
EXTERN_C_BEGIN
extern void LAPACKgetri_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,
                         PetscBLASInt*,PetscScalar*,PetscBLASInt*,
                         PetscBLASInt*);
EXTERN_C_END
