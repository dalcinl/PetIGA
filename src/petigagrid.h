#if !defined(PETIGAGRID_H)
#define PETIGAGRID_H

#include <petsc.h>

#if !defined(LGMap)
#define LGMap ISLocalToGlobalMapping
#endif

typedef struct _n_IGA_Grid *IGA_Grid;

struct _n_IGA_Grid {
  MPI_Comm    comm;
  PetscInt    dim,dof;
  PetscInt    sizes[3];
  PetscInt    local_start[3];
  PetscInt    local_width[3];
  PetscInt    ghost_start[3];
  PetscInt    ghost_width[3];
  AO          ao;
  LGMap       lgmap;
  PetscLayout map;
  Vec         lvec,gvec,nvec;
  VecScatter  g2l,l2g,g2n;
};

PETSC_EXTERN PetscErrorCode IGA_Grid_Create(MPI_Comm,IGA_Grid*);
PETSC_EXTERN PetscErrorCode IGA_Grid_Init(IGA_Grid,
                                          PetscInt,PetscInt,
                                          const PetscInt[],
                                          const PetscInt[],
                                          const PetscInt[],
                                          const PetscInt[],
                                          const PetscInt[]);
PETSC_EXTERN PetscErrorCode IGA_Grid_Reset(IGA_Grid);
PETSC_EXTERN PetscErrorCode IGA_Grid_Destroy(IGA_Grid*);
PETSC_EXTERN PetscErrorCode IGA_Grid_LocalIndices(IGA_Grid,PetscInt,PetscInt*,PetscInt*[]);
PETSC_EXTERN PetscErrorCode IGA_Grid_GhostIndices(IGA_Grid,PetscInt,PetscInt*,PetscInt*[]);
PETSC_EXTERN PetscErrorCode IGA_Grid_SetAO(IGA_Grid,AO);
PETSC_EXTERN PetscErrorCode IGA_Grid_GetAO(IGA_Grid,AO*);
PETSC_EXTERN PetscErrorCode IGA_Grid_SetLGMap(IGA_Grid,LGMap);
PETSC_EXTERN PetscErrorCode IGA_Grid_GetLGMap(IGA_Grid,LGMap*);
PETSC_EXTERN PetscErrorCode IGA_Grid_GetLayout(IGA_Grid,PetscLayout*);
PETSC_EXTERN PetscErrorCode IGA_Grid_GetVecLocal  (IGA_Grid,const VecType,Vec*);
PETSC_EXTERN PetscErrorCode IGA_Grid_GetVecGlobal (IGA_Grid,const VecType,Vec*);
PETSC_EXTERN PetscErrorCode IGA_Grid_GetVecNatural(IGA_Grid,const VecType,Vec*);
PETSC_EXTERN PetscErrorCode IGA_Grid_GetScatterG2L(IGA_Grid,VecScatter*);
PETSC_EXTERN PetscErrorCode IGA_Grid_GetScatterL2G(IGA_Grid,VecScatter*);
PETSC_EXTERN PetscErrorCode IGA_Grid_GetScatterG2N(IGA_Grid,VecScatter*);

PETSC_EXTERN PetscErrorCode IGA_Grid_GlobalToLocal(IGA_Grid,Vec,Vec);
PETSC_EXTERN PetscErrorCode IGA_Grid_LocalToGlobal(IGA_Grid,Vec,Vec,InsertMode);
PETSC_EXTERN PetscErrorCode IGA_Grid_NaturalToGlobal(IGA_Grid,Vec,Vec);
PETSC_EXTERN PetscErrorCode IGA_Grid_GlobalToNatural(IGA_Grid,Vec,Vec);

PETSC_EXTERN PetscErrorCode IGA_Grid_NewScatterApp(IGA_Grid g,
                                                   const PetscInt[],
                                                   const PetscInt[],
                                                   const PetscInt[],
                                                   const PetscInt[],
                                                   Vec*,VecScatter*,VecScatter*);

#endif/*PETIGAGRID_H*/
