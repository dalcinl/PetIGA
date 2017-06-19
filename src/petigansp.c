#include "petiga.h"

EXTERN_C_BEGIN
extern PetscReal IGA_Greville(PetscInt i,PetscInt p,const PetscReal U[]);
EXTERN_C_END

EXTERN_C_BEGIN
extern void IGA_LobattoPoints(PetscInt n,PetscReal x0,PetscReal x1,PetscReal x[]);
EXTERN_C_END

PETSC_STATIC_INLINE
PetscReal ComputePoint(PetscInt index,IGAAxis axis,IGABasisType btype)
{
  if (PetscUnlikely(axis->p == 0)) return 0;
  if (btype == IGA_BASIS_SPECTRAL) {
    PetscInt n = axis->nel;
    PetscInt p = axis->p;
    PetscInt e = index / p;
    PetscInt i = index % p;
    PetscReal u0,u1,u[16+1];
    if (PetscUnlikely(p > 16)) return PETSC_MAX_REAL;
    if (PetscUnlikely(e == n)) { e -= 1; i = p; }
    u0 = axis->U[axis->span[e]];
    u1 = axis->U[axis->span[e]+1];
    IGA_LobattoPoints(p+1,u0,u1,u);
    return u[i];
  }
  return IGA_Greville(index,axis->p,axis->U);
}

PETSC_STATIC_INLINE PetscInt Product(const PetscInt a[3]) { return a[0]*a[1]*a[2]; }

PetscErrorCode IGACreateCoordinates(IGA iga,Vec *coords)
{
  MPI_Comm       comm;
  PetscInt       dim,nsd,n,N;
  Vec            vecX;
  PetscScalar    *arrayX;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(coords,2);
  IGACheckSetUpStage2(iga,1);

  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = IGAGetGeometryDim(iga,&nsd);CHKERRQ(ierr);
  dim = PetscClipInterval(dim,1,3);
  nsd = PetscClipInterval(nsd,dim,3);
  n = nsd*Product(iga->node_lwidth);
  N = nsd*Product(iga->node_sizes);

  ierr = VecCreate(comm,&vecX);CHKERRQ(ierr);
  ierr = VecSetSizes(vecX,n,N);CHKERRQ(ierr);
  ierr = VecSetBlockSize(vecX,nsd);CHKERRQ(ierr);
  ierr = VecSetType(vecX,iga->vectype);CHKERRQ(ierr);
  *coords = vecX;

  ierr = VecGetArray(vecX,&arrayX);CHKERRQ(ierr);
  if (iga->geometry && iga->geometryX) {
    /* local non-ghosted grid */
    const PetscInt *lstart = iga->node_lstart;
    const PetscInt *lwidth = iga->node_lwidth;
    PetscInt ilstart = lstart[0], ilend = lstart[0]+lwidth[0];
    PetscInt jlstart = lstart[1], jlend = lstart[1]+lwidth[1];
    PetscInt klstart = lstart[2], klend = lstart[2]+lwidth[2];
    /* local ghosted grid */
    const PetscInt *gstart = iga->node_gstart;
    const PetscInt *gwidth = iga->node_gwidth;
    PetscInt igstart = gstart[0], igend = gstart[0]+gwidth[0];
    PetscInt jgstart = gstart[1], jgend = gstart[1]+gwidth[1];
    PetscInt kgstart = gstart[2], kgend = gstart[2]+gwidth[2];
    /* fill coordinates using control points */
    PetscInt c,i,j,k;
    PetscInt xpos = 0,index = 0;
    PetscReal *xyz = iga->geometryX;
    for (k=kgstart; k<kgend; k++)
      for (j=jgstart; j<jgend; j++)
        for (i=igstart; i<igend; i++, index++)
          if (i>=ilstart && i<ilend &&
              j>=jlstart && j<jlend &&
              k>=klstart && k<klend)
            {
              for (c=0; c<nsd; c++)
                arrayX[xpos++] = xyz[index+c];
            }
  } else {
    /* local non-ghosted grid */
    const PetscInt *shift  = iga->node_shift;
    const PetscInt *lstart = iga->node_lstart;
    const PetscInt *lwidth = iga->node_lwidth;
    PetscInt ilstart = lstart[0], ilend = lstart[0]+lwidth[0];
    PetscInt jlstart = lstart[1], jlend = lstart[1]+lwidth[1];
    PetscInt klstart = lstart[2], klend = lstart[2]+lwidth[2];
    /* fill coordinates using Greville abscissae */
    IGAAxis  *AX = iga->axis;
    IGABasis *BD = iga->basis;
    PetscInt c,i,j,k,pos = 0;
    PetscReal uvw[3] = {0,0,0};
    for (k=klstart; k<klend; k++) {
      uvw[2] = ComputePoint(k+shift[2],AX[2],BD[2]->type);
      for (j=jlstart; j<jlend; j++) {
        uvw[1] = ComputePoint(j+shift[1],AX[1],BD[1]->type);
        for (i=ilstart; i<ilend; i++) {
          uvw[0] = ComputePoint(i+shift[0],AX[0],BD[0]->type);
          {
            for (c=0; c<nsd; c++)
              arrayX[pos++] = uvw[c];
          }
        }
      }
    }
  }
  ierr = VecRestoreArray(vecX,&arrayX);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode IGACreateRigidBody(IGA iga,MatNullSpace *nsp)
{
  Vec            coords;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(nsp,2);
  IGACheckSetUpStage2(iga,1);
  ierr = IGACreateCoordinates(iga,&coords);CHKERRQ(ierr);
  ierr = MatNullSpaceCreateRigidBody(coords,nsp);CHKERRQ(ierr);
  ierr = VecDestroy(&coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
