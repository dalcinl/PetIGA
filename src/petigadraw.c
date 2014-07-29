#include "petigaprobe.h"

#if PETSC_VERSION_LE(3,3,0)
#define DMSetCoordinates DMDASetCoordinates
#define DMGetCoordinates DMDAGetCoordinates
#endif

PETSC_EXTERN PetscErrorCode IGACreateDrawDM(IGA iga,PetscInt bs,DM *dm);
PETSC_EXTERN PetscErrorCode IGAGetDrawDM(IGA iga,DM *dm);

#undef  __FUNCT__
#define __FUNCT__ "IGACreateDrawDM"
PetscErrorCode IGACreateDrawDM(IGA iga,PetscInt bs,DM *dm)
{
  MPI_Comm        comm;
  PetscInt        i,dim;
  PetscInt        sizes[3] = {1, 1, 1};
  PetscInt        width[3] = {1, 1, 1};
  PetscBool       wraps[3] = {PETSC_TRUE, PETSC_TRUE, PETSC_TRUE};
  PetscInt        n,N;
  Vec             X;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  PetscValidPointer(dm,2);
  IGACheckSetUpStage1(iga,1);

  /* compute global and local sizes */
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  if (!iga->collocation) {
    const PetscInt *pranks = iga->proc_ranks;
    const PetscInt *psizes = iga->proc_sizes;
    for (i=0; i<dim; i++) {
      sizes[i] = iga->elem_sizes[i]*iga->axis[i]->p + 1;
      width[i] = iga->elem_width[i]*iga->axis[i]->p + (pranks[i] == psizes[i]-1);
    }
  } else {
    for (i=0; i<dim; i++) {
      sizes[i] = iga->node_sizes[i];
      width[i] = iga->node_lwidth[i];
    }
  }
  /* create DMDA context */
  ierr = IGACreateDMDA(iga,bs,sizes,width,wraps,PETSC_TRUE,1,dm);CHKERRQ(ierr);
  /* create coordinate vector */
  n = width[0]*width[1]*width[2];
  N = sizes[0]*sizes[1]*sizes[2];
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = VecCreate(comm,&X);CHKERRQ(ierr);
  ierr = VecSetSizes(X,dim*n,dim*N);CHKERRQ(ierr);
  ierr = VecSetBlockSize(X,dim);CHKERRQ(ierr);
  ierr = VecSetType(X,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecSetUp(X);CHKERRQ(ierr);
  ierr = DMSetCoordinates(*dm,X);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetDrawDM"
PetscErrorCode IGAGetDrawDM(IGA iga,DM *dm)
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(dm,2);
  IGACheckSetUpStage2(iga,1);
  if (!iga->draw_dm) {
    ierr = IGACreateDrawDM(iga,iga->dof,&iga->draw_dm);CHKERRQ(ierr);
    if (iga->fieldname)
      for (i=0; i<iga->dof; i++)
        {ierr = DMDASetFieldName(iga->draw_dm,i,iga->fieldname[i]);CHKERRQ(ierr);}
  }
  *dm = iga->draw_dm;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscReal IGA_Greville(PetscInt i,PetscInt p,const PetscReal U[]);
EXTERN_C_END

PETSC_STATIC_INLINE
PetscReal GrevilleParameter(PetscInt index,IGAAxis axis)
{
  if (PetscUnlikely(axis->p == 0)) return 0.0;
  return IGA_Greville(index,axis->p,axis->U);
}

PETSC_STATIC_INLINE
PetscReal LagrangeParameter(PetscInt index,IGAAxis axis)
{
  PetscInt p,e,i,k;
  PetscReal u0,u1;
  if (PetscUnlikely(axis->p == 0)) return 0.0;
  p = axis->p;
  e = index / p;
  i = index % p;
  if (PetscUnlikely(e == axis->nel))
    return axis->U[axis->span[e-1] + 1];
  k  = axis->span[e];
  u0 = axis->U[k];
  u1 = axis->U[k+1];
  return u0 + i*(u1-u0)/p;
}

#undef  __FUNCT__
#define __FUNCT__ "IGADrawVec"
PetscErrorCode IGADrawVec(IGA iga,Vec vec,PetscViewer viewer)
{
  PetscInt       dim,dof;
  DM             da;
  Vec            X,U;
  PetscScalar    *arrayX=NULL;
  PetscScalar    *arrayU=NULL;
  IGAProbe       probe;
  PetscReal     (*Parameter)(PetscInt,IGAAxis);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,3);
  PetscCheckSameComm(iga,1,vec,2);
  PetscCheckSameComm(iga,1,viewer,3);
  IGACheckSetUp(iga,1);

  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);

  ierr = IGAGetDrawDM(iga,&da);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&X);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da,&U);CHKERRQ(ierr);
  ierr = VecGetArray(X,&arrayX);CHKERRQ(ierr);
  ierr = VecGetArray(U,&arrayU);CHKERRQ(ierr);

  Parameter = iga->collocation ? GrevilleParameter : LagrangeParameter;

  ierr = IGAProbeCreate(iga,vec,&probe);CHKERRQ(ierr);
  ierr = IGAProbeSetOrder(probe,0);CHKERRQ(ierr);
  ierr = IGAProbeSetCollective(probe,PETSC_FALSE);CHKERRQ(ierr);
  {
    PetscReal uvw[3] = {0.0, 0.0, 0.0};
    PetscReal xval[3];
    PetscScalar *uval;
    PetscInt is,iw,js,jw,ks,kw;
    PetscInt c,i,j,k,xpos=0,upos=0;
    ierr = DMDAGetCorners(da,&is,&js,&ks,&iw,&jw,&kw);CHKERRQ(ierr);
    ierr = PetscMalloc1(dof,&uval);CHKERRQ(ierr);
    for (k=ks; k<ks+kw; k++) {
      uvw[2] = Parameter(k,iga->axis[2]);
      for (j=js; j<js+jw; j++) {
        uvw[1] = Parameter(j,iga->axis[1]);
        for (i=is; i<is+iw; i++) {
          uvw[0] = Parameter(i,iga->axis[0]);
          {
            ierr = IGAProbeSetPoint(probe,uvw);CHKERRQ(ierr);
            ierr = IGAProbeGeomMap(probe,xval);CHKERRQ(ierr);
            ierr = IGAProbeFormValue(probe,uval);CHKERRQ(ierr);
            for (c=0; c<dim; c++) arrayX[xpos++] = xval[c];
            for (c=0; c<dof; c++) arrayU[upos++] = uval[c];
          }
        }
      }
    }
    ierr = PetscFree(uval);CHKERRQ(ierr);
  }
  ierr = IGAProbeDestroy(&probe);CHKERRQ(ierr);

  ierr = VecRestoreArray(X,&arrayX);CHKERRQ(ierr);
  ierr = VecRestoreArray(U,&arrayU);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject)U,((PetscObject)vec)->name);CHKERRQ(ierr);
  ierr = VecView(U,viewer);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)U,NULL);CHKERRQ(ierr);

  ierr = DMRestoreGlobalVector(da,&U);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
