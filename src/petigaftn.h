#if !defined(PETIGAFTN_H)
#define PETIGAFTN_H

#include <petsc.h>

EXTERN_C_BEGIN
extern void IGA_Quadrature_1D(PetscInt,const PetscReal[],const PetscReal[],const PetscReal*,
                              PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_Quadrature_2D(PetscInt,const PetscReal[],const PetscReal[],const PetscReal*,
                              PetscInt,const PetscReal[],const PetscReal[],const PetscReal*,
                              PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_Quadrature_3D(PetscInt,const PetscReal[],const PetscReal[],const PetscReal*,
                              PetscInt,const PetscReal[],const PetscReal[],const PetscReal*,
                              PetscInt,const PetscReal[],const PetscReal[],const PetscReal*,
                              PetscReal[],PetscReal[],PetscReal[]);
EXTERN_C_END


EXTERN_C_BEGIN
extern void IGA_BasisFuns_1D(PetscInt,
                             PetscInt,PetscInt,const PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_BasisFuns_2D(PetscInt,
                             PetscInt,PetscInt,const PetscReal[],
                             PetscInt,PetscInt,const PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_BasisFuns_3D(PetscInt,
                             PetscInt,PetscInt,const PetscReal[],
                             PetscInt,PetscInt,const PetscReal[],
                             PetscInt,PetscInt,const PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
EXTERN_C_END


EXTERN_C_BEGIN
extern void IGA_Rationalize_1D(PetscInt,PetscInt,PetscInt,const PetscReal[],
                               PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_Rationalize_2D(PetscInt,PetscInt,PetscInt,const PetscReal[],
                               PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_Rationalize_3D(PetscInt,PetscInt,PetscInt,const PetscReal[],
                               PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
EXTERN_C_END


EXTERN_C_BEGIN
extern void IGA_GeometryMap_1D(PetscInt,PetscInt,PetscInt,const PetscReal[],
                               const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                               PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_GeometryMap_2D(PetscInt,PetscInt,PetscInt,const PetscReal[],
                               const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                               PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_GeometryMap_3D(PetscInt,PetscInt,PetscInt,const PetscReal[],
                               const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                               PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_GeometryMap   (PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,const PetscReal[],
                               const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                               PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
EXTERN_C_END


EXTERN_C_BEGIN
extern void IGA_InverseMap_1D(PetscInt,PetscInt,
                              const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                              PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_InverseMap_2D(PetscInt,PetscInt,
                              const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                              PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_InverseMap_3D(PetscInt,PetscInt,
                              const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                              PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
EXTERN_C_END


EXTERN_C_BEGIN
extern void IGA_ShapeFuns_1D(PetscInt,PetscInt,PetscInt,
                             const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                             const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_ShapeFuns_2D(PetscInt,PetscInt,PetscInt,
                             const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                             const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_ShapeFuns_3D(PetscInt,PetscInt,PetscInt,
                             const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                             const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
EXTERN_C_END

#endif/*PETIGAFTN_H*/
