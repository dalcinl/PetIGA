#include "petigagrid.h"
#include <petsc-private/petscimpl.h>

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_Create"
PetscErrorCode IGA_Grid_Create(MPI_Comm comm,IGA_Grid *grid)
{
  IGA_Grid       g;
  PetscInt       i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(grid,9);

  ierr = PetscMalloc(sizeof(**grid),grid);CHKERRQ(ierr);
  ierr = PetscMemzero(*grid,sizeof(**grid));CHKERRQ(ierr);
  g = *grid;

  g->comm = comm;
  for (i=0; i<3; i++) {
    g->sizes[i]       = 1;
    g->local_start[i] = 0;
    g->local_width[i] = 1;
    g->ghost_start[i] = 0;
    g->ghost_width[i] = 1;
  }

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_Init"
PetscErrorCode IGA_Grid_Init(IGA_Grid g,
                             PetscInt dim,PetscInt dof,
                             const PetscInt sizes[],
                             const PetscInt local_start[],
                             const PetscInt local_width[],
                             const PetscInt ghost_start[],
                             const PetscInt ghost_width[])
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidIntPointer(sizes,4);
  PetscValidIntPointer(local_start,5);
  PetscValidIntPointer(local_width,6);
  PetscValidIntPointer(ghost_start,7);
  PetscValidIntPointer(ghost_width,8);

  ierr = IGA_Grid_Reset(g);CHKERRQ(ierr);
  g->dim = dim;
  g->dof = dof;
  for (i=0; i<dim; i++) {
    g->sizes[i]       = sizes[i];
    g->local_start[i] = local_start[i];
    g->local_width[i] = local_width[i];
    g->ghost_start[i] = ghost_start[i];
    g->ghost_width[i] = ghost_width[i];
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_Reset"
PetscErrorCode IGA_Grid_Reset(IGA_Grid g)
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!g) PetscFunctionReturn(0);
  PetscValidPointer(g,1);
  for (i=0; i<3; i++) {
    g->sizes[i]       = 1;
    g->local_start[i] = 0;
    g->local_width[i] = 1;
    g->ghost_start[i] = 0;
    g->ghost_width[i] = 1;
  }
  ierr = AODestroy(&g->ao);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&g->lgmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&g->lgmapb);CHKERRQ(ierr);
  ierr = VecDestroy(&g->lvec);CHKERRQ(ierr);
  ierr = VecDestroy(&g->gvec);CHKERRQ(ierr);
  ierr = VecDestroy(&g->nvec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&g->g2l);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&g->l2g);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&g->g2n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_Destroy"
PetscErrorCode IGA_Grid_Destroy(IGA_Grid *grid)
{
  IGA_Grid       g;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!grid) PetscFunctionReturn(0);
  PetscValidPointer(grid,1);
  PetscValidPointer(*grid,1);
  g = *grid; *grid = NULL;
  ierr = IGA_Grid_Reset(g);CHKERRQ(ierr);
  ierr = PetscFree(g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_LocalIndices"
PetscErrorCode IGA_Grid_LocalIndices(IGA_Grid g,PetscInt bs,PetscInt *nlocal,PetscInt *indices[])
{
  PetscInt       nloc;
  PetscInt       *iloc;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidIntPointer(nlocal,3);
  PetscValidPointer(indices,4);
  {
    const PetscInt *sizes = g->sizes;
    const PetscInt *lstart = g->local_start;
    const PetscInt *lwidth = g->local_width;
    /* global grid strides */
    PetscInt jstride = sizes[0];
    PetscInt kstride = sizes[0]*sizes[1];
    /* local non-ghosted grid */
    PetscInt ilstart = lstart[0], ilend = lstart[0]+lwidth[0];
    PetscInt jlstart = lstart[1], jlend = lstart[1]+lwidth[1];
    PetscInt klstart = lstart[2], klend = lstart[2]+lwidth[2];
    PetscInt c,i,j,k,pos = 0;
    nloc = bs*lwidth[0]*lwidth[1]*lwidth[2];
    ierr = PetscMalloc(nloc*sizeof(PetscInt),&iloc);CHKERRQ(ierr);
    for (k=klstart; k<klend; k++)
      for (j=jlstart; j<jlend; j++)
        for (i=ilstart; i<ilend; i++)
          for (c=0; c<bs; c++)
            iloc[pos++] = c + bs*(i + j * jstride + k * kstride);
  }
  *nlocal  = nloc;
  *indices = iloc;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_GhostIndices"
PetscErrorCode IGA_Grid_GhostIndices(IGA_Grid g,PetscInt bs,PetscInt *nghost,PetscInt *indices[])
{
  PetscInt       nght;
  PetscInt       *ight;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidIntPointer(nghost,3);
  PetscValidPointer(indices,4);
  {
    const PetscInt *sizes = g->sizes;
    const PetscInt *gstart = g->ghost_start;
    const PetscInt *gwidth = g->ghost_width;
    /* global grid */
    PetscInt isize = sizes[0]/*istride = 1*/;
    PetscInt jsize = sizes[1], jstride = isize;
    PetscInt ksize = sizes[2], kstride = isize*jsize;
    /* local ghosted grid */
    PetscInt igstart = gstart[0], igend = gstart[0]+gwidth[0];
    PetscInt jgstart = gstart[1], jgend = gstart[1]+gwidth[1];
    PetscInt kgstart = gstart[2], kgend = gstart[2]+gwidth[2];
    /* compute local ghosted indices in global natural numbering */
    PetscInt c,i,j,k,pos = 0;
    nght = bs*gwidth[0]*gwidth[1]*gwidth[2];
    ierr = PetscMalloc(nght*sizeof(PetscInt),&ight);CHKERRQ(ierr);
    for (k=kgstart; k<kgend; k++)
      for (j=jgstart; j<jgend; j++)
        for (i=igstart; i<igend; i++) {
          PetscInt ig = i, jg = j, kg = k; /* account for periodicicty */
          if (ig<0) ig = isize + ig; else if (ig>=isize) ig = ig % isize;
          if (jg<0) jg = jsize + jg; else if (jg>=jsize) jg = jg % jsize;
          if (kg<0) kg = ksize + kg; else if (kg>=ksize) kg = kg % ksize;
          for (c=0; c<bs; c++)
            ight[pos++] = c + bs*(ig + jg * jstride + kg * kstride);
        }
  }
  *nghost  = nght;
  *indices = ight;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_SetAO"
PetscErrorCode IGA_Grid_SetAO(IGA_Grid g,AO ao)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidHeaderSpecific(ao,AO_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)ao);CHKERRQ(ierr);
  ierr = AODestroy(&g->ao);CHKERRQ(ierr);
  g->ao = ao;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_GetAO"
PetscErrorCode IGA_Grid_GetAO(IGA_Grid g,AO *ao)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidPointer(ao,2);
  if (!g->ao) {
    PetscInt napp,*iapp;
    ierr = IGA_Grid_LocalIndices(g,1,&napp,&iapp);CHKERRQ(ierr);
    ierr = AOCreateMemoryScalable(g->comm,napp,iapp,NULL,&g->ao);CHKERRQ(ierr);
    ierr = PetscFree(iapp);CHKERRQ(ierr);
  }
  *ao = g->ao;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_SetLGMapBlock"
PetscErrorCode IGA_Grid_SetLGMapBlock(IGA_Grid g,LGMap lgmapb)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidHeaderSpecific(lgmapb,IS_LTOGM_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)lgmapb);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&g->lgmapb);CHKERRQ(ierr);
  g->lgmapb = lgmapb;
  ierr = ISLocalToGlobalMappingDestroy(&g->lgmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_GetLGMapBlock"
PetscErrorCode IGA_Grid_GetLGMapBlock(IGA_Grid g,LGMap *lgmapb)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidPointer(lgmapb,2);
  if (!g->lgmapb) {
    PetscInt nghost,*ighost;
    ierr = IGA_Grid_GhostIndices(g,1,&nghost,&ighost);CHKERRQ(ierr);
    ierr = IGA_Grid_GetAO(g,&g->ao);CHKERRQ(ierr);
    ierr = AOApplicationToPetsc(g->ao,nghost,ighost);CHKERRQ(ierr);
#if PETSC_VERSION_LT(3,5,0)
    ierr = ISLocalToGlobalMappingCreate(g->comm,nghost,ighost,PETSC_OWN_POINTER,&g->lgmapb);CHKERRQ(ierr);
#else
    ierr = ISLocalToGlobalMappingCreate(g->comm,g->dof,nghost,ighost,PETSC_OWN_POINTER,&g->lgmapb);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)g->lgmapb);CHKERRQ(ierr);
    g->lgmap = g->lgmapb;
#endif
  }
  *lgmapb = g->lgmapb;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_GetLGMap"
PetscErrorCode IGA_Grid_GetLGMap(IGA_Grid g,LGMap *lgmap)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidPointer(lgmap,2);
  if (!g->lgmapb) {ierr = IGA_Grid_GetLGMapBlock(g,&g->lgmapb);CHKERRQ(ierr);}
  if (!g->lgmap) {
#if PETSC_VERSION_LT(3,5,0)
    ierr = ISLocalToGlobalMappingUnBlock(g->lgmapb,g->dof,&g->lgmap);CHKERRQ(ierr);
    if (g->lgmapb != g->lgmap)
      {ierr = PetscObjectCompose((PetscObject)g->lgmap,"__IGA_lgmapb",(PetscObject)g->lgmapb);CHKERRQ(ierr);}
#else
    ierr = PetscObjectReference((PetscObject)g->lgmapb);CHKERRQ(ierr);
    g->lgmap = g->lgmapb;
#endif
  }
  *lgmap = g->lgmap;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_GetLayout"
PetscErrorCode IGA_Grid_GetLayout(IGA_Grid g,PetscLayout *map)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidPointer(map,2);
  if (!g->map) {
    LGMap lgmap;
    const PetscInt *sizes = g->sizes;
    const PetscInt *width = g->local_width;
    PetscInt bs = g->dof;
    PetscInt n  = bs*width[0]*width[1]*width[2];
    PetscInt N  = bs*sizes[0]*sizes[1]*sizes[2];
    ierr = PetscLayoutCreate(g->comm,&g->map);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(g->map,bs);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(g->map,n);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(g->map,N);CHKERRQ(ierr);
    ierr = IGA_Grid_GetLGMap(g,&lgmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetISLocalToGlobalMapping(g->map,lgmap);CHKERRQ(ierr);
#if PETSC_VERSION_LT(3,5,0)
    ierr = IGA_Grid_GetLGMapBlock(g,&lgmap);CHKERRQ(ierr);
    ierr = PetscLayoutSetISLocalToGlobalMappingBlock(g->map,lgmap);CHKERRQ(ierr);
#endif
    ierr = PetscLayoutSetUp(g->map);CHKERRQ(ierr);
  }

  *map = g->map;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_GetVecLocal"
PetscErrorCode IGA_Grid_GetVecLocal(IGA_Grid g,const VecType vtype,Vec *lvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  if (vtype) PetscValidCharPointer(vtype,2);
  PetscValidPointer(lvec,3);
  if (!g->lvec) {
    const PetscInt *width = g->ghost_width;
    PetscInt n  = width[0]*width[1]*width[2];
    PetscInt bs = g->dof;
    ierr = VecCreate(PETSC_COMM_SELF,&g->lvec);CHKERRQ(ierr);
    ierr = VecSetSizes(g->lvec,n*bs,n*bs);CHKERRQ(ierr);
    ierr = VecSetBlockSize(g->lvec,bs);CHKERRQ(ierr);
    ierr = VecSetType(g->lvec,vtype?vtype:VECSTANDARD);CHKERRQ(ierr);
  }
  *lvec = g->lvec;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_GetVecGlobal"
PetscErrorCode IGA_Grid_GetVecGlobal(IGA_Grid g,const VecType vtype,Vec *gvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  if (vtype) PetscValidCharPointer(vtype,2);
  PetscValidPointer(gvec,3);
  if (!g->gvec) {
    const PetscInt *sizes = g->sizes;
    const PetscInt *width = g->local_width;
    PetscInt n  = width[0]*width[1]*width[2];
    PetscInt N  = sizes[0]*sizes[1]*sizes[2];
    PetscInt bs = g->dof;
    ierr = VecCreate(g->comm,&g->gvec);CHKERRQ(ierr);
    ierr = VecSetSizes(g->gvec,n*bs,N*bs);CHKERRQ(ierr);
    ierr = VecSetBlockSize(g->gvec,bs);CHKERRQ(ierr);
    ierr = VecSetType(g->gvec,vtype?vtype:VECSTANDARD);CHKERRQ(ierr);
  }
  *gvec = g->gvec;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_GetVecNatural"
PetscErrorCode IGA_Grid_GetVecNatural(IGA_Grid g,const VecType vtype,Vec *nvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  if (vtype) PetscValidCharPointer(vtype,2);
  PetscValidPointer(nvec,3);
  if (!g->nvec) {
    Vec gvec;
    ierr = IGA_Grid_GetVecGlobal(g,vtype,&gvec);CHKERRQ(ierr);
    ierr = VecDuplicate(gvec,&g->nvec);CHKERRQ(ierr);
  }
  *nvec = g->nvec;
  PetscFunctionReturn(0);
}

#if PETSC_VERSION_LT(3,5,0)
#define ISCreateBlock(comm,bs,n,idx,mode,is) \
        ISCreateBlock(comm,bs,n,idx,((mode)==PETSC_USE_POINTER)?PETSC_COPY_VALUES:(mode),is)
#endif

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_GetScatterG2L"
PetscErrorCode IGA_Grid_GetScatterG2L(IGA_Grid g,VecScatter *g2l)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidPointer(g2l,2);
  if (!g->g2l) {
    LGMap lgmap;
    IS isghost;
    Vec gvec,lvec;
    PetscInt nghost;
    const PetscInt *ighost;
#if PETSC_VERSION_LT(3,5,0)
    ierr = IGA_Grid_GetLGMapBlock(g,&lgmap);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(lgmap,&ighost);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(lgmap,&nghost);CHKERRQ(ierr);
#else
    ierr = IGA_Grid_GetLGMap(g,&lgmap);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetBlockIndices(lgmap,&ighost);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(lgmap,&nghost);CHKERRQ(ierr);
    nghost /= g->dof;
#endif
    ierr = IGA_Grid_GetVecGlobal(g,VECSTANDARD,&gvec);CHKERRQ(ierr);
    ierr = IGA_Grid_GetVecLocal(g,VECSTANDARD,&lvec);CHKERRQ(ierr);
    ierr = ISCreateBlock(g->comm,g->dof,nghost,ighost,PETSC_USE_POINTER,&isghost);CHKERRQ(ierr);
    ierr = VecScatterCreate(gvec,isghost,lvec,NULL,&g->g2l);CHKERRQ(ierr);
    ierr = ISDestroy(&isghost);CHKERRQ(ierr);
#if PETSC_VERSION_LT(3,5,0)
    ierr = ISLocalToGlobalMappingRestoreIndices(lgmap,&ighost);CHKERRQ(ierr);
#else
    ierr = ISLocalToGlobalMappingRestoreBlockIndices(lgmap,&ighost);CHKERRQ(ierr);
#endif
  }
  *g2l = g->g2l;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_GetScatterL2G"
PetscErrorCode IGA_Grid_GetScatterL2G(IGA_Grid g,VecScatter *l2g)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidPointer(l2g,2);
  if (!g->l2g) {
    IS isglobal,islocal;
    Vec gvec,lvec;
    /* local non-ghosted grid */
    const PetscInt *lstart = g->local_start;
    const PetscInt *lwidth = g->local_width;
    PetscInt ilstart = lstart[0], ilend = lstart[0]+lwidth[0];
    PetscInt jlstart = lstart[1], jlend = lstart[1]+lwidth[1];
    PetscInt klstart = lstart[2], klend = lstart[2]+lwidth[2];
    /* local ghosted grid */
    const PetscInt *gstart = g->ghost_start;
    const PetscInt *gwidth = g->ghost_width;
    PetscInt igstart = gstart[0], igend = gstart[0]+gwidth[0];
    PetscInt jgstart = gstart[1], jgend = gstart[1]+gwidth[1];
    PetscInt kgstart = gstart[2], kgend = gstart[2]+gwidth[2];
    /* */
    PetscInt nlocal = lwidth[0]*lwidth[1]*lwidth[2],*ilocal,start;
    PetscInt i,j,k,pos = 0,index = 0;
    ierr = PetscMalloc(nlocal*sizeof(PetscInt),&ilocal);CHKERRQ(ierr);
    for (k=kgstart; k<kgend; k++)
      for (j=jgstart; j<jgend; j++)
        for (i=igstart; i<igend; i++, index++)
          if (i>=ilstart && i<ilend &&
              j>=jlstart && j<jlend &&
              k>=klstart && k<klend)
            ilocal[pos++] = index;
    /* */
    ierr = IGA_Grid_GetVecGlobal(g,VECSTANDARD,&gvec);CHKERRQ(ierr);
    ierr = IGA_Grid_GetVecLocal(g,VECSTANDARD,&lvec);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(gvec,&start,NULL);CHKERRQ(ierr);
    ierr = ISCreateBlock(PETSC_COMM_SELF,g->dof,nlocal,ilocal,PETSC_OWN_POINTER,&islocal);CHKERRQ(ierr);
    ierr = ISCreateStride(g->comm,nlocal*g->dof,start,1,&isglobal);CHKERRQ(ierr);
    ierr = VecScatterCreate(lvec,islocal,gvec,isglobal,&g->l2g);CHKERRQ(ierr);
    ierr = ISDestroy(&islocal);CHKERRQ(ierr);
    ierr = ISDestroy(&isglobal);CHKERRQ(ierr);
  }
  *l2g = g->l2g;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_GetScatterG2N"
PetscErrorCode IGA_Grid_GetScatterG2N(IGA_Grid g,VecScatter *g2n)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidPointer(g2n,2);
  if (!g->g2n) {
    IS isnatural,isglobal;
    Vec gvec,nvec;
    PetscInt nlocal,*inatural,start;
    ierr = IGA_Grid_GetVecGlobal (g,VECSTANDARD,&gvec);CHKERRQ(ierr);
    ierr = IGA_Grid_GetVecNatural(g,VECSTANDARD,&nvec);CHKERRQ(ierr);
    ierr = IGA_Grid_LocalIndices(g,1,&nlocal,&inatural);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(gvec,&start,NULL);CHKERRQ(ierr);
    ierr = ISCreateStride(g->comm,nlocal*g->dof,start,1,&isglobal);CHKERRQ(ierr);
    ierr = ISCreateBlock(g->comm,g->dof,nlocal,inatural,PETSC_OWN_POINTER,&isnatural);CHKERRQ(ierr);
    ierr = VecScatterCreate(gvec,isglobal,nvec,isnatural,&g->g2n);CHKERRQ(ierr);
    ierr = ISDestroy(&isglobal);CHKERRQ(ierr);
    ierr = ISDestroy(&isnatural);CHKERRQ(ierr);
  }
  *g2n = g->g2n;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_GlobalToLocal"
PetscErrorCode IGA_Grid_GlobalToLocal(IGA_Grid g,Vec gvec,Vec lvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,3);
  ierr = IGA_Grid_GetScatterG2L(g,&g->g2l);CHKERRQ(ierr);
  ierr = VecScatterBegin(g->g2l,gvec,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (g->g2l,gvec,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_LocalToGlobal"
PetscErrorCode IGA_Grid_LocalToGlobal(IGA_Grid g,Vec lvec,Vec gvec,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,3);
  if (addv == ADD_VALUES) {
    ierr = IGA_Grid_GetScatterG2L(g,&g->g2l);CHKERRQ(ierr);
    ierr = VecScatterBegin(g->g2l,lvec,gvec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (g->g2l,lvec,gvec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  } else if (addv == INSERT_VALUES) {
    ierr = IGA_Grid_GetScatterL2G(g,&g->l2g);CHKERRQ(ierr);
    ierr = VecScatterBegin(g->l2g,lvec,gvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (g->l2g,lvec,gvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  } else SETERRQ(g->comm,PETSC_ERR_SUP,"Not yet implemented");
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_NaturalToGlobal"
PetscErrorCode IGA_Grid_NaturalToGlobal(IGA_Grid g,Vec nvec,Vec gvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidHeaderSpecific(nvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,3);
  ierr = IGA_Grid_GetScatterG2N(g,&g->g2n);CHKERRQ(ierr);
  ierr = VecScatterBegin(g->g2n,nvec,gvec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (g->g2n,nvec,gvec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_GlobalToNatural"
PetscErrorCode IGA_Grid_GlobalToNatural(IGA_Grid g,Vec gvec,Vec nvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(nvec,VEC_CLASSID,3);
  ierr = IGA_Grid_GetScatterG2N(g,&g->g2n);CHKERRQ(ierr);
  ierr = VecScatterBegin(g->g2n,gvec,nvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (g->g2n,gvec,nvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_NewScatterApp"
PetscErrorCode IGA_Grid_NewScatterApp(IGA_Grid g,
                                      const PetscInt sizes[],
                                      const PetscInt start[],
                                      const PetscInt width[],
                                      Vec        *avec,
                                      VecScatter *a2g,
                                      VecScatter *g2a)
{
  const char*    vtype = VECSTANDARD;
  Vec            gvec;
  Vec            nvec;
  VecScatter     n2g;
  VecScatter     g2n;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(g,1);
  PetscValidPointer(avec,2);
  PetscValidPointer(a2g,3);
  PetscValidPointer(g2a,4);
  /* global vector */
  {
    ierr = IGA_Grid_GetVecGlobal(g,vtype,&gvec);CHKERRQ(ierr);
    ierr = VecGetType(gvec,&vtype);CHKERRQ(ierr);
  }
  /* natural vector */
  {
    PetscInt n  = width[0]*width[1]*width[2];
    PetscInt N  = sizes[0]*sizes[1]*sizes[2];
    PetscInt bs = g->dof;
    ierr = VecCreate(g->comm,&nvec);CHKERRQ(ierr);
    ierr = VecSetSizes(nvec,n*bs,N*bs);CHKERRQ(ierr);
    ierr = VecSetBlockSize(nvec,bs);CHKERRQ(ierr);
    ierr = VecSetType(nvec,vtype);CHKERRQ(ierr);
  }
  /* natural -> global scatter */
  {
    IS isnatural,isglobal;
    PetscInt gstart,*inatural;
    /* global grid strides */
    PetscInt jstride = sizes[0];
    PetscInt kstride = sizes[0]*sizes[1];
    /* local non-ghosted grid */
    const PetscInt *lstart = g->local_start;
    const PetscInt *lwidth = g->local_width;
    PetscInt ilstart = lstart[0], ilend = lstart[0]+lwidth[0];
    PetscInt jlstart = lstart[1], jlend = lstart[1]+lwidth[1];
    PetscInt klstart = lstart[2], klend = lstart[2]+lwidth[2];
    /* */
    PetscInt nlocal = lwidth[0]*lwidth[1]*lwidth[2];
    PetscInt i,j,k,pos = 0;
    ierr = VecGetOwnershipRange(gvec,&gstart,NULL);CHKERRQ(ierr);
    ierr = PetscMalloc(nlocal*sizeof(PetscInt),&inatural);CHKERRQ(ierr);
    for (k=klstart; k<klend; k++)
      for (j=jlstart; j<jlend; j++)
        for (i=ilstart; i<ilend; i++)
          inatural[pos++] = i + j * jstride + k * kstride;
    ierr = ISCreateBlock(g->comm,g->dof,nlocal,inatural,PETSC_OWN_POINTER,&isnatural);CHKERRQ(ierr);
    ierr = ISCreateStride(g->comm,nlocal*g->dof,gstart,1,&isglobal);CHKERRQ(ierr);
    ierr = VecScatterCreate(nvec,isnatural,gvec,isglobal,&n2g);CHKERRQ(ierr);
    ierr = ISDestroy(&isnatural);CHKERRQ(ierr);
    ierr = ISDestroy(&isglobal);CHKERRQ(ierr);
  }
  /* global -> natural scatter */
  {
    LGMap lgmap;
    IS isglobal,isnatural;
    PetscInt *iglobal,*inatural;
    /* local non-ghosted grid */
    PetscInt istart = start[0], iend = start[0]+width[0]/*istride = 1*/;
    PetscInt jstart = start[1], jend = start[1]+width[1], jstride = sizes[0];
    PetscInt kstart = start[2], kend = start[2]+width[2], kstride = sizes[0]*sizes[1];
    /* local ghosted grid */
    const PetscInt *gstart = g->ghost_start;
    const PetscInt *gwidth = g->ghost_width;
    PetscInt igstart = gstart[0], igend = gstart[0]+gwidth[0];
    PetscInt jgstart = gstart[1], jgend = gstart[1]+gwidth[1];
    PetscInt kgstart = gstart[2], kgend = gstart[2]+gwidth[2];
    /* */
    PetscInt nlocal = width[0]*width[1]*width[2];
    PetscInt i,j,k,pos = 0,index = 0;
    ierr = PetscMalloc(nlocal*sizeof(PetscInt),&iglobal);CHKERRQ(ierr);
    ierr = PetscMalloc(nlocal*sizeof(PetscInt),&inatural);CHKERRQ(ierr);
    for (k=kgstart; k<kgend; k++)
      for (j=jgstart; j<jgend; j++)
        for (i=igstart; i<igend; i++, index++)
          if (i>=istart && i<iend &&
              j>=jstart && j<jend &&
              k>=kstart && k<kend)
            {
              iglobal [pos] = index;
              inatural[pos] = i + j * jstride + k * kstride;
              pos++;
            }
#if PETSC_VERSION_LT(3,5,0)
    ierr = IGA_Grid_GetLGMapBlock(g,&lgmap);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApply(lgmap,nlocal,iglobal,iglobal);CHKERRQ(ierr);
#else
    ierr = IGA_Grid_GetLGMap(g,&lgmap);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApplyBlock(lgmap,nlocal,iglobal,iglobal);CHKERRQ(ierr);
#endif
    ierr = ISCreateBlock(g->comm,g->dof,nlocal,iglobal,PETSC_OWN_POINTER,&isglobal);CHKERRQ(ierr);
    ierr = ISCreateBlock(g->comm,g->dof,nlocal,inatural,PETSC_OWN_POINTER,&isnatural);CHKERRQ(ierr);
    ierr = VecScatterCreate(gvec,isglobal,nvec,isnatural,&g2n);CHKERRQ(ierr);
    ierr = ISDestroy(&isnatural);CHKERRQ(ierr);
    ierr = ISDestroy(&isglobal);CHKERRQ(ierr);
  }

  *avec = nvec;
  *a2g  = n2g;
  *g2a  = g2n;
  PetscFunctionReturn(0);
}
