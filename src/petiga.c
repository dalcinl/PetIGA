#include "petiga.h"

static PetscErrorCode DMSetMatType(DM,const MatType);
#include "private/matimpl.h"
#include "private/dmimpl.h"
#undef  __FUNCT__
#define __FUNCT__ "DMSetMatType"
static
PetscErrorCode DMSetMatType(DM dm,const MatType mattype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscFree(dm->mattype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(mattype,&dm->mattype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreate"
PetscErrorCode IGACreate(MPI_Comm comm,IGA *_iga)
{
  PetscInt       i;
  IGA            iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_iga,2);
  *_iga = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = IGAInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  ierr = PetscHeaderCreate(iga,_p_IGA,struct _IGAOps,IGA_CLASSID,-1,
                           "IGA","IGA","IGA",comm,IGADestroy,IGAView);CHKERRQ(ierr);
  *_iga = iga;

  ierr = PetscNew(struct _IGAUserOps,&iga->userops);CHKERRQ(ierr);
  iga->vectype = PETSC_NULL;
  iga->mattype = PETSC_NULL;

  iga->dim = -1;
  iga->dof = -1;

  for (i=0; i<3; i++) {
    ierr = IGAAxisCreate(&iga->axis[i] );CHKERRQ(ierr);
    ierr = IGARuleCreate(&iga->rule[i]);CHKERRQ(ierr);
    ierr = IGABasisCreate(&iga->basis[i]);CHKERRQ(ierr);
    ierr = IGABoundaryCreate(&iga->boundary[i][0]);CHKERRQ(ierr);
    ierr = IGABoundaryCreate(&iga->boundary[i][1]);CHKERRQ(ierr);
  }
  ierr = IGAElementCreate(&iga->iterator);CHKERRQ(ierr);

  iga->rational = PETSC_FALSE;
  iga->geometry = PETSC_FALSE;
  iga->dm_geom  = 0;
  iga->dm_dof   = 0;

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGADestroy"
PetscErrorCode IGADestroy(IGA *_iga)
{
  PetscInt       i;
  IGA            iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_iga,1);
  iga = *_iga; *_iga = 0;
  if (!iga) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (--((PetscObject)iga)->refct > 0) PetscFunctionReturn(0);;

  ierr = PetscFree(iga->userops);CHKERRQ(ierr);
  ierr = PetscFree(iga->vectype);CHKERRQ(ierr);
  ierr = PetscFree(iga->mattype);CHKERRQ(ierr);
  for (i=0; i<3; i++) {
    ierr = IGAAxisDestroy(&iga->axis[i]);CHKERRQ(ierr);
    ierr = IGARuleDestroy(&iga->rule[i]);CHKERRQ(ierr);
    ierr = IGABasisDestroy(&iga->basis[i]);CHKERRQ(ierr);
    ierr = IGABoundaryDestroy(&iga->boundary[i][0]);CHKERRQ(ierr);
    ierr = IGABoundaryDestroy(&iga->boundary[i][1]);CHKERRQ(ierr);
  }
  ierr = IGAElementDestroy(&iga->iterator);CHKERRQ(ierr);

  ierr = DMDestroy(&iga->dm_geom);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->dm_dof);CHKERRQ(ierr);

  ierr = PetscHeaderDestroy(&iga);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAView"
PetscErrorCode IGAView(IGA iga,PetscViewer viewer)
{
  PetscBool      isstring;
  PetscBool      isascii;
  PetscBool      isbinary;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)iga)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(iga,1,viewer,2);
  if (!iga->setup) PetscFunctionReturn(0);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isascii) PetscFunctionReturn(0);
  {
    PetscInt i;
    MPI_Comm comm;
    PetscInt dim,dof;
    PetscInt procs[3],nodes[3];
    PetscInt start[3],width[3];
    ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);
    ierr = DMDAGetInfo(iga->dm_dof,0,
                       &nodes[0],&nodes[1],&nodes[2],
                       &procs[0],&procs[1],&procs[2],
                       0,0,0,0,0,0);CHKERRQ(ierr);
    ierr = DMDAGetCorners(iga->dm_dof,
                          &start[0],&start[1],&start[2],
                          &width[0],&width[1],&width[2]);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"IGA: dimension=%D  dofs/node=%D\n",dim,dof);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      IGAAxis  *AX = iga->axis;
      IGARule  *QR = iga->rule;
      IGABasis *BD = iga->basis;
      PetscViewerASCIIPrintf(viewer,"Axis %D: periodic=%d  degree=%D  quadrature=%D  processors=%D  nodes=%D  elements=%D\n",
                             i,(int)AX[i]->periodic,AX[i]->p,QR[i]->nqp,procs[i],nodes[i],BD[i]->nel);CHKERRQ(ierr);
    }
    { /* */
      PetscInt nnodes=1,nelems=1;
      PetscInt icount[2],isum[2],imin[2],imax[2];
      for (i=0; i<dim; i++) {
        PetscInt s = iga->iterator->range[i][0];
        PetscInt e = iga->iterator->range[i][1];
        nnodes *= width[i];
        nelems *= e - s;
      }
      icount[0] = nnodes;
      icount[1] = nelems;
      ierr = MPI_Allreduce(&icount,&isum,2,MPIU_INT,MPIU_SUM,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&icount,&imin,2,MPIU_INT,MPIU_MIN,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&icount,&imax,2,MPIU_INT,MPIU_MAX,comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Partitioning - nodes:    sum=%D  min=%D  max=%D  max/min=%g\n",
                                    isum[0],imin[0],imax[0],(double)imax[0]/(double)imin[0]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Partitioning - elements: sum=%D  min=%D  max=%D  max/min=%g\n",
                                    isum[1],imin[1],imax[1],(double)imax[1]/(double)imin[1]);CHKERRQ(ierr);
    }
    /*
    PetscMPIInt index = 0;
    PetscInt ranks[3] = {1,1,1};
    ierr = MPI_Comm_rank(comm,&index);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      ranks[i] = index % procs[i];
      index -= ranks[i];
      index /= procs[i];
    }
    ierr = MPI_Comm_rank(((PetscObject)iga)->comm,&index);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,
                                              "[%d] (%D,%D,%D): ",
                                              (int)index,ranks[0],ranks[1],ranks[2]);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"nodes=%D elements=%D\n",nnodes,nelems);
    ierr = PetscViewerFlush(viewer);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);
    */
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetComm"
PetscErrorCode IGAGetComm(IGA iga,MPI_Comm *comm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(comm,2);
  *comm = ((PetscObject)iga)->comm;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetDim"
PetscErrorCode IGASetDim(IGA iga,PetscInt dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,dim,2);
  if (iga->dim > 0 && iga->dim != dim)
    SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
             "Cannot change IGA dim from %D after it was set to %D",iga->dim,dim);
  iga->dim = dim;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetDim"
PetscErrorCode IGAGetDim(IGA iga,PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(dim,2);
  *dim = iga->dim;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetDof"
PetscErrorCode IGASetDof(IGA iga,PetscInt dof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,dof,2);
  if (iga->dof > 0 && iga->dof != dof)
    SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
             "Cannot change IGA dof from %D after it was set to %D",iga->dof,dof);
  iga->dof = dof;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetDof"
PetscErrorCode IGAGetDof(IGA iga,PetscInt *dof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(dof,2);
  *dof = iga->dof;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetAxis"
PetscErrorCode IGAGetAxis(IGA iga,PetscInt i,IGAAxis *axis)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(axis,3);
  if (iga->dim <= 0) SETERRQ (((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetDim() first");
  if (i < 0)         SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Axis index %D must be nonnegative",i);
  if (i >= iga->dim) SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Axis index %D, but dim %D",i,iga->dim);
  *axis = iga->axis[i];
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetRule"
PetscErrorCode IGAGetRule(IGA iga,PetscInt i,IGARule *rule)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(rule,3);
  if (iga->dim <= 0) SETERRQ (((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetDim() first");
  if (i < 0)         SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D must be nonnegative",i);
  if (i >= iga->dim) SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D, but dim %D",i,iga->dim);
  *rule = iga->rule[i];
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetBoundary"
PetscErrorCode IGAGetBoundary(IGA iga,PetscInt i,PetscInt side,IGABoundary *boundary)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(boundary,4);
  if (iga->dim <= 0) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetDim() first");
  if (iga->dof <= 0) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetDof() first");
  if (i < 0)         SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D must be nonnegative",i);
  if (i >= iga->dim) SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D, but dimension %D",i,iga->dim);
  if (side < 0) side = 0; /* XXX error ?*/
  if (side > 1) side = 1; /* XXX error ?*/
  if (iga->boundary[i][side]->dof < 1) {
    ierr = IGABoundaryInit(iga->boundary[i][side],iga->dof);CHKERRQ(ierr);
  }
  *boundary = iga->boundary[i][side];
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetFromOptions"
PetscErrorCode IGASetFromOptions(IGA iga)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  {
    PetscBool flg;
    char vtype[256] = VECSTANDARD;
    char mtype[256] = MATBAIJ;
    if (iga->dof < 2) {ierr= PetscStrcpy(mtype,MATAIJ);CHKERRQ(ierr);}
    if (iga->vectype) {ierr= PetscStrcpy(vtype,iga->vectype);CHKERRQ(ierr);}
    if (iga->mattype) {ierr= PetscStrcpy(mtype,iga->mattype);CHKERRQ(ierr);}
    ierr = PetscObjectOptionsBegin((PetscObject)iga);CHKERRQ(ierr);
    ierr = PetscOptionsList("-iga_vec_type","Vector type","VecSetType",VecList,vtype,vtype,sizeof vtype,&flg);CHKERRQ(ierr);
    if (flg) {ierr= IGASetVecType(iga,vtype);CHKERRQ(ierr);}
    ierr = PetscOptionsList("-iga_mat_type","Matrix type","MatSetType",MatList,mtype,mtype,sizeof mtype,&flg);CHKERRQ(ierr);
    if (flg) {ierr= IGASetMatType(iga,mtype);CHKERRQ(ierr);}
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)iga);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateDM"
static PetscErrorCode IGACreateDM(IGA iga,PetscInt dof,DM *_dm)
{
  PetscInt         i,dim;
  PetscInt         procs[3]   = {-1,-1,-1};
  PetscInt         sizes[3]   = { 1, 1, 1};
  const PetscInt   *ranges[3] = { 0, 0, 0};
  PetscInt         swidth = 0;
  DMDABoundaryType btype[3] = {DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE};
  DM               dm = 0, dm_base = 0;
  PetscErrorCode   ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(_dm,3);
  *_dm = PETSC_NULL;

  if (!dm_base) {ierr = IGAGetDofDM (iga,&dm_base);CHKERRQ(ierr);}
  if (!dm_base) {ierr = IGAGetGeomDM(iga,&dm_base);CHKERRQ(ierr);}
  if ( dm_base) {PetscValidHeaderSpecific(iga,DM_CLASSID,0);}

  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  if (dm_base) {
    ierr = DMDAGetInfo(dm_base,0,
                       &sizes[0],&sizes[1],&sizes[2],
                       &procs[0],&procs[1],&procs[2],0,
                       &swidth,&btype[0],&btype[1],&btype[2],0);CHKERRQ(ierr);
    ierr = DMDAGetOwnershipRanges(dm_base,&ranges[0],&ranges[1],&ranges[2]);CHKERRQ(ierr);
  } else {
    for (i=0; i<dim; i++) {
      IGAAxis   axis;
      PetscBool periodic;
      PetscInt  p,n,m;
      ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
      ierr = IGAAxisGetPeriodic(axis,&periodic);CHKERRQ(ierr);
      ierr = IGAAxisGetOrder(axis,&p);CHKERRQ(ierr);
      ierr = IGAAxisGetKnots(axis,&m,0);CHKERRQ(ierr);
      n = m-p-1;
      swidth = PetscMax(swidth,p);
      sizes[i] = periodic ? n+1-p : n+1;
      btype[i] = periodic ? DMDA_BOUNDARY_PERIODIC : DMDA_BOUNDARY_NONE;
    }
  }

  ierr = DMDACreate(((PetscObject)iga)->comm,&dm);CHKERRQ(ierr);
  ierr = DMDASetDim(dm,dim); CHKERRQ(ierr);
  ierr = DMDASetDof(dm,dof); CHKERRQ(ierr);
  ierr = DMDASetSizes(dm,sizes[0],sizes[1],sizes[2]); CHKERRQ(ierr);
  ierr = DMDASetNumProcs(dm,procs[0],procs[1],procs[2]);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(dm,ranges[0],ranges[1],ranges[2]);CHKERRQ(ierr);
  ierr = DMDASetStencilType(dm,DMDA_STENCIL_BOX); CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(dm,swidth); CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(dm,btype[0],btype[1],btype[2]); CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  *_dm = dm;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateDofDM"
static PetscErrorCode IGACreateDofDM(IGA iga,DM *dm_dof)
{
  PetscInt         dof;
  PetscErrorCode   ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(dm_dof,2);
  ierr = IGAGetDof(iga,&dof); CHKERRQ(ierr);
  ierr = IGACreateDM(iga,dof,dm_dof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateGeomDM"
static PetscErrorCode IGACreateGeomDM(IGA iga,DM *dm_geom)
{
  PetscInt       dim;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(dm_geom,2);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = IGACreateDM(iga,dim+1,dm_geom);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetUp"
PetscErrorCode IGASetUp(IGA iga)
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (iga->setup) PetscFunctionReturn(0);

  if (iga->dim < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
            "Must call IGASetDim() first");

  if (iga->dof < 1)
    iga->dof = 1;

  for (i=0; i<3; i++) {
    if (!iga->axis[i])  {ierr = IGAAxisCreate(&iga-> axis[i]);CHKERRQ(ierr);}
    if (!iga->rule[i])  {ierr = IGARuleCreate(&iga->rule[i]);CHKERRQ(ierr);}
    if (!iga->basis[i]) {ierr = IGABasisCreate(&iga->basis[i]);CHKERRQ(ierr);}
  }
  if (!iga->iterator) {
    ierr = IGAElementCreate(&iga->iterator);CHKERRQ(ierr);
  }

  for (i=0; i<iga->dim; i++) {
    ierr = IGAAxisCheck(iga->axis[i]);CHKERRQ(ierr);
  }
  for (i=0; i<3; i++) {
    if (!iga->rule[i]->nqp) {
      PetscInt p = iga->axis[i]->p;
      PetscInt q = p+1;
      ierr = IGARuleInit(iga->rule[i],q);CHKERRQ(ierr);
    }
    if (!iga->basis[i]->nel) {
      PetscInt p = iga->axis[i]->p;
      PetscInt d = PetscMin(p,3); /* XXX */
      ierr = IGABasisInit(iga->basis[i],iga->axis[i],iga->rule[i],d);CHKERRQ(ierr);
    }
  }

  if (!iga->vectype) {
    const MatType vtype = VECSTANDARD;
    ierr = PetscStrallocpy(vtype,&iga->vectype);CHKERRQ(ierr);
  }
  if (!iga->mattype) {
    const MatType mtype = (iga->dof > 1) ? MATBAIJ : MATAIJ;
    ierr = PetscStrallocpy(mtype,&iga->mattype);CHKERRQ(ierr);
  }

  ierr = IGACreateDofDM(iga,&iga->dm_dof);CHKERRQ(ierr);
  ierr = DMSetVecType(iga->dm_dof,iga->vectype);CHKERRQ(ierr);
  ierr = DMSetMatType(iga->dm_dof,iga->mattype);CHKERRQ(ierr);
  /*ierr = DMSetOptionsPrefix(iga->dm_dof, "dof_"); CHKERRQ(ierr);*/
  /*ierr = DMSetFromOptions(iga->dm_dof); CHKERRQ(ierr);*/

  iga->setup = PETSC_TRUE;

  iga->iterator->parent = iga;
  ierr = IGAElementSetUp(iga->iterator);CHKERRQ(ierr);

  { /* */
    PetscBool flg;
    char filename[PETSC_MAX_PATH_LEN] = "";
    PetscViewer viewer;
    ierr = PetscOptionsGetString(((PetscObject)iga)->prefix,"-iga_view",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg && !PetscPreLoadingOn) {
      ierr = PetscViewerASCIIOpen(((PetscObject)iga)->comm,filename,&viewer);CHKERRQ(ierr);
      ierr = IGAView(iga,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetDofDM"
PetscErrorCode IGAGetDofDM(IGA iga, DM *dm_dof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(dm_dof,2);
  *dm_dof = iga->dm_dof;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetGeomDM"
PetscErrorCode IGAGetGeomDM(IGA iga, DM *dm_geom)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(dm_geom,2);
  *dm_geom = iga->dm_geom;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetVecType"
PetscErrorCode IGASetVecType(IGA iga,const VecType vectype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidCharPointer(vectype,2);
  ierr = PetscFree(iga->vectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(vectype,&iga->vectype);CHKERRQ(ierr);
  if (iga->dm_dof) {
    ierr = DMSetVecType(iga->dm_dof,iga->vectype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#undef  __FUNCT__
#define __FUNCT__ "IGASetMatType"
PetscErrorCode IGASetMatType(IGA iga,const MatType mattype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidCharPointer(mattype,2);
  ierr = PetscFree(iga->mattype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(mattype,&iga->mattype);CHKERRQ(ierr);
  if (iga->dm_dof) {
    ierr = DMSetMatType(iga->dm_dof,iga->mattype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateVec"
PetscErrorCode IGACreateVec(IGA iga, Vec *vec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(vec,2);
  ierr = DMCreateGlobalVector(iga->dm_dof,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if PETSC_VERSION_(3,2,0)
#define DMCreateMatrix DMGetMatrix
#endif

#undef  __FUNCT__
#define __FUNCT__ "IGACreateMat"
PetscErrorCode IGACreateMat(IGA iga, Mat *mat)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(mat,2);
  ierr = DMCreateMatrix(iga->dm_dof,iga->mattype,mat);CHKERRQ(ierr);
  ierr = MatSetOption(*mat,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  /*ierr = MatSetOption(*mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);*/
  {
    PetscInt bs;
    ierr = DMGetBlockSize(iga->dm_dof,&bs);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize((*mat)->rmap,bs);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize((*mat)->cmap,bs);CHKERRQ(ierr);
  }
  {
    ISLocalToGlobalMapping ltog,ltogb;
    ierr = DMGetLocalToGlobalMapping(iga->dm_dof,&ltog);CHKERRQ(ierr);
    ierr = DMGetLocalToGlobalMappingBlock(iga->dm_dof,&ltogb);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(*mat,ltog,ltog);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMappingBlock(*mat,ltogb,ltogb);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetLocalVec"
PetscErrorCode IGAGetLocalVec(IGA iga,Vec *lvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(lvec,2);
  ierr = DMGetLocalVector(iga->dm_dof,lvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGARestoreLocalVec"
PetscErrorCode IGARestoreLocalVec(IGA iga,Vec *lvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(lvec,2);
  PetscValidHeaderSpecific(*lvec,VEC_CLASSID,2);
  ierr = DMRestoreLocalVector(iga->dm_dof,lvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateVecGlobal"
PetscErrorCode IGAGetGlobalVec(IGA iga,Vec *gvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(gvec,2);
  ierr = DMGetGlobalVector(iga->dm_dof,gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGARestoreGlobalVec"
PetscErrorCode IGARestoreGlobalVec(IGA iga,Vec *gvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(gvec,2);
  PetscValidHeaderSpecific(*gvec,VEC_CLASSID,2);
  ierr = DMRestoreGlobalVector(iga->dm_dof,gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGlobalToLocal"
PetscErrorCode IGAGlobalToLocal(IGA iga,Vec gvec,Vec lvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,3);
  ierr = DMGlobalToLocalBegin(iga->dm_dof,gvec,INSERT_VALUES,lvec);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (iga->dm_dof,gvec,INSERT_VALUES,lvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGALocalToGlobal"
PetscErrorCode IGALocalToGlobal(IGA iga,Vec lvec,Vec gvec,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,3);
  ierr = DMLocalToGlobalBegin(iga->dm_dof,lvec,addv,gvec);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (iga->dm_dof,lvec,addv,gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetElement"
PetscErrorCode IGAGetElement(IGA iga,IGAElement *element)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(element,2);
  *element = iga->iterator;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetUserSystem"
PetscErrorCode IGASetUserSystem(IGA iga,IGAUserSystem System,void *SysCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (System) iga->userops->System = System;
  if (SysCtx) iga->userops->SysCtx = SysCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetUserFunction"
PetscErrorCode IGASetUserFunction(IGA iga,IGAUserFunction Function,void *FunCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (Function) iga->userops->Function = Function;
  if (FunCtx)   iga->userops->FunCtx   = FunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetUserJacobian"
PetscErrorCode IGASetUserJacobian(IGA iga,IGAUserJacobian Jacobian,void *JacCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (Jacobian) iga->userops->Jacobian = Jacobian;
  if (JacCtx)   iga->userops->JacCtx   = JacCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetUserIFunction"
PetscErrorCode IGASetUserIFunction(IGA iga,IGAUserIFunction IFunction,void *IFunCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (IFunction) iga->userops->IFunction = IFunction;
  if (IFunCtx)   iga->userops->IFunCtx   = IFunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetUserIJacobian"
PetscErrorCode IGASetUserIJacobian(IGA iga,IGAUserIJacobian IJacobian,void *IJacCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (IJacobian) iga->userops->IJacobian = IJacobian;
  if (IJacCtx)   iga->userops->IJacCtx   = IJacCtx;
  PetscFunctionReturn(0);
}
