#include "petiga.h"

EXTERN_C_BEGIN
extern PetscReal IGA_Greville(PetscInt i,PetscInt p,const PetscReal U[]);
EXTERN_C_END

PETSC_STATIC_INLINE PetscInt Product(const PetscInt a[3]) { return a[0]*a[1]*a[2]; }

#undef  __FUNCT__
#define __FUNCT__ "IGACreateCoordinates"
PetscErrorCode IGACreateCoordinates(IGA iga,Vec *coords)
{
  MPI_Comm       comm;
  PetscInt       dim,n,N;
  Vec            vecX;
  PetscScalar    *arrayX;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(coords,2);
  IGACheckSetUpStage2(iga,1);

  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = IGAGetGeometryDim(iga,&dim);CHKERRQ(ierr);
  if (!dim) {ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);}
  n = dim*Product(iga->node_lwidth);
  N = dim*Product(iga->node_sizes);

  ierr = VecCreate(comm,&vecX);CHKERRQ(ierr);
  ierr = VecSetSizes(vecX,n,N);CHKERRQ(ierr);
  ierr = VecSetBlockSize(vecX,dim);CHKERRQ(ierr);
  ierr = VecSetType(vecX,iga->vectype);CHKERRQ(ierr);
  *coords = vecX;

  ierr = VecGetArray(vecX,&arrayX);CHKERRQ(ierr);
  if (iga->geometry) {
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
              for (c=0; c<dim; c++)
                arrayX[xpos++] = xyz[index+c];
            }
  } else {
    /* local non-ghosted grid */
    const PetscInt *lstart = iga->node_lstart;
    const PetscInt *lwidth = iga->node_lwidth;
    PetscInt ilstart = lstart[0], ilend = lstart[0]+lwidth[0];
    PetscInt jlstart = lstart[1], jlend = lstart[1]+lwidth[1];
    PetscInt klstart = lstart[2], klend = lstart[2]+lwidth[2];
    /* fill coordinates using Greville abscissae */
    IGAAxis *AX = iga->axis;
    PetscInt c,i,j,k;
    PetscInt xpos = 0;
    PetscReal xyz[3] = {0.0, 0.0, 0.0};
    for (k=klstart; k<klend; k++) {
      xyz[2] = IGA_Greville(k,AX[2]->p,AX[2]->U);
      for (j=jlstart; j<jlend; j++) {
        xyz[1] = IGA_Greville(j,AX[1]->p,AX[1]->U);
        for (i=ilstart; i<ilend; i++) {
          xyz[0] = IGA_Greville(i,AX[0]->p,AX[0]->U);
          {
            for (c=0; c<dim; c++)
              arrayX[xpos++] = xyz[c];
          }
        }
      }
    }
  }
  ierr = VecRestoreArray(vecX,&arrayX);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateRigidBody"
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
