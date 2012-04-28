#if !defined(PETIGAGRID_H)
#define PETIGAGRID_H

#include <petsc.h>
PETSC_EXTERN_CXX_BEGIN

#ifndef LGMap
#define LGMap ISLocalToGlobalMapping
#endif

typedef struct _n_IGA_Grid *IGA_Grid;

struct _n_IGA_Grid {
  MPI_Comm   comm;
  PetscInt   dim,dof;
  PetscInt   sizes[3];
  PetscInt   local_start[3];
  PetscInt   local_width[3];
  PetscInt   ghost_start[3];
  PetscInt   ghost_width[3];
  AO         ao,aob;
  LGMap      lgmap,lgmapb;
  Vec        gvec,lvec;
  VecScatter g2l,l2g,g2n;
};

PetscErrorCode IGA_Grid_Create(MPI_Comm,IGA_Grid*);
PetscErrorCode IGA_Grid_Init(IGA_Grid,
                             PetscInt,PetscInt,
                             const PetscInt[],
                             const PetscInt[],
                             const PetscInt[],
                             const PetscInt[],
                             const PetscInt[]);
PetscErrorCode IGA_Grid_Reset(IGA_Grid);
PetscErrorCode IGA_Grid_Destroy(IGA_Grid*);
PetscErrorCode IGA_Grid_LocalIndices(IGA_Grid,PetscInt,PetscInt*,PetscInt*[]);
PetscErrorCode IGA_Grid_GhostIndices(IGA_Grid,PetscInt,PetscInt*,PetscInt*[]);
PetscErrorCode IGA_Grid_SetAOBlock(IGA_Grid,AO);
PetscErrorCode IGA_Grid_GetAOBlock(IGA_Grid,AO*);
PetscErrorCode IGA_Grid_GetAO(IGA_Grid,AO*);
PetscErrorCode IGA_Grid_SetLGMapBlock(IGA_Grid,LGMap);
PetscErrorCode IGA_Grid_GetLGMapBlock(IGA_Grid,LGMap*);
PetscErrorCode IGA_Grid_GetLGMap(IGA_Grid,LGMap*);
PetscErrorCode IGA_Grid_GetVecGlobal(IGA_Grid,const VecType,Vec*);
PetscErrorCode IGA_Grid_GetVecLocal (IGA_Grid,const VecType,Vec*);
PetscErrorCode IGA_Grid_GetScatterG2L(IGA_Grid,VecScatter*);
PetscErrorCode IGA_Grid_GetScatterL2G(IGA_Grid,VecScatter*);
PetscErrorCode IGA_Grid_GetScatterG2N(IGA_Grid,VecScatter*);

PETSC_EXTERN_CXX_END
#endif/*PETIGAGRID_H*/
