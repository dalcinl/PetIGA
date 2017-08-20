#include "petiga.h"

#include <petsclog.h>
#include <petsc/private/dmdaimpl.h>

#if PETSC_VERSION_LT(3,8,0)
#define PC_MG_GALERKIN_BOTH PETSC_TRUE
#endif

static
PetscErrorCode DMDASetCoarseningFactor(DM da,PetscInt coarsen_x,PetscInt coarsen_y,PetscInt coarsen_z)
{
  DM_DA *dd = (DM_DA*)da->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidLogicalCollectiveInt(da,coarsen_x,2);
  PetscValidLogicalCollectiveInt(da,coarsen_y,3);
  PetscValidLogicalCollectiveInt(da,coarsen_z,4);
  if (coarsen_x > 0) dd->coarsen_x = coarsen_x;
  if (coarsen_y > 0) dd->coarsen_y = coarsen_y;
  if (coarsen_z > 0) dd->coarsen_z = coarsen_z;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDAComputeCoarsenLevels(DM,PetscInt*);
static PetscErrorCode DMDAComputeCoarsenFactor(DM);
static PetscErrorCode DMDACoarsenHook_PCMG(DM,DM,void*);

static const PetscInt primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                                  31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                                  73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                                  127, 131, 137, 139, 149, 151, 157, 163, 167, 173};
static const PetscInt nprimes  = (PetscInt)(sizeof(primes)/sizeof(PetscInt));

PETSC_STATIC_INLINE
PetscBool CoarsenFactor(PetscInt dim,const PetscInt M[3],const PetscInt P[3],PetscInt factor[3])
{
  PetscInt tiny = 2*3; /* */
  PetscInt i,j;
  for (i=0; i<dim; i++) factor[i] = M[i];
  /* If the grid is small enough, stop coarsening */
  if (M[0]*M[1]*M[2] <= 1) /* XXX */
    return PETSC_FALSE;
  if (M[0] <= tiny && M[1] <= tiny && M[2] <= tiny)
    return PETSC_FALSE;
  for (i=0; i<dim; i++)
    if (M[i] <= P[i])
      return PETSC_FALSE;
  /* Compute minimum possible coarsening for each direction */
  for (i=0; i<dim; i++)
    for (j=0; j<nprimes; j++)
      if (M[i] % primes[j] == 0)
        {factor[i] = primes[j]; break;}
  /* Avoid coarsening a direction if any of the others is large */
  for (i=0; i<dim; i++)
    for (j=0; j<dim; j++)
      if (i != j && M[i] <= M[j]/factor[j])
        factor[i] = 1;
  return PETSC_TRUE;
}

static
PetscErrorCode DMDACoarsenHook_PCMG(DM dm,DM dmc,void *ctx)
{(void)dm; (void)ctx; return DMDAComputeCoarsenFactor(dmc);}

static
PetscErrorCode DMDAComputeCoarsenLevels(DM dm,PetscInt *outlevels)
{
  PetscInt       i,dim,M[3],P[3];
  DMBoundaryType btype[3];
  PetscInt       factor[3] = {1,1,1};
  PetscInt       levels = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidIntPointer(outlevels,2);
  ierr = DMDAGetInfo(dm,&dim,&M[0],&M[1],&M[2],&P[0],&P[1],&P[2],NULL,NULL,&btype[0],&btype[1],&btype[2],NULL);CHKERRQ(ierr);
  for (i=0; i<dim; i++) if (btype[i] == DM_BOUNDARY_NONE) M[i] -= 1;
  while (CoarsenFactor(dim,M,P,factor) && levels < 64)
    {for (i=0; i<dim; i++) M[i] /= factor[i]; levels++;}
  *outlevels = levels;
  PetscFunctionReturn(0);
}

static
PetscErrorCode DMDAComputeCoarsenFactor(DM dm)
{
  PetscInt       i,dim,M[3],P[3];
  DMBoundaryType btype[3];
  PetscInt       factor[3] = {1,1,1};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMDAGetInfo(dm,&dim,&M[0],&M[1],&M[2],&P[0],&P[1],&P[2],NULL,NULL,&btype[0],&btype[1],&btype[2],NULL);CHKERRQ(ierr);
  for (i=0; i<dim; i++) if (btype[i] == DM_BOUNDARY_NONE) M[i] -= 1;
  (void)CoarsenFactor(dim,M,P,factor);
  ierr = DMDASetCoarseningFactor(dm,factor[0],factor[1],factor[2]);CHKERRQ(ierr);
  ierr = DMCoarsenHookAdd(dm,DMDACoarsenHook_PCMG,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscInfo6(dm,"DA dimensions (%3D,%3D,%3D) coarsen factors (%3D,%3D,%3D)\n",M[0],M[1],M[2],factor[0],factor[1],factor[2]);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm,NULL,"-mg_levels_da_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPreparePCMG(IGA iga,PC pc)
{
  PetscBool      match,set;
  const char     *prefix = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(pc,PC_CLASSID,2);

  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCMG,&match);CHKERRQ(ierr);
  if (!match) PetscFunctionReturn(0);

  {
    DM        da;
    PetscBool wraps[3] = {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE};
    PetscInt  i,dim,dof,*N = iga->node_sizes,*n = iga->node_lwidth;
    PetscInt  levels;
    /* Use the Galerkin process to compute coarse-level operators */
    ierr = PetscOptionsHasName(((PetscObject)pc)->options,prefix,"-pc_mg_galerkin",&set);CHKERRQ(ierr);
    if (!set) {ierr = PCMGSetGalerkin(pc,PC_MG_GALERKIN_BOTH);CHKERRQ(ierr);}
    /* Honor -pc_mg_levels 1 explicitly passed in the command line */
    ierr = PetscOptionsHasName(((PetscObject)pc)->options,prefix,"-pc_mg_levels",&set);CHKERRQ(ierr);
    ierr = PCMGGetLevels(pc,&levels);CHKERRQ(ierr);
    if (set && levels == 1) PetscFunctionReturn(0);
    /* Use a DMDA to generate the grid hierarchy with low-order levels */
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);
    for (i=0; i<dim; i++) wraps[i] = iga->axis[i]->periodic;
    ierr = IGACreateDMDA(iga,dof,N,n,wraps,PETSC_TRUE,1,&da);CHKERRQ(ierr);
    ierr = DMSetOptionsPrefix(da,prefix);CHKERRQ(ierr);
    ierr = PCSetDM(pc,da);CHKERRQ(ierr);
    /* Compute number of multigrid levels */
    if (levels <= 1) {
      ierr = DMDAComputeCoarsenLevels(da,&levels);CHKERRQ(ierr);
      ierr = PCMGSetLevels(pc,levels,NULL);CHKERRQ(ierr);
    }
    ierr = DMDAComputeCoarsenFactor(da);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
