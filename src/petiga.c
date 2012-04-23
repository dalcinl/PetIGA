#include "petiga.h"

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
    iga->proc_sizes[i] = -1;
  }
  ierr = IGAElementCreate(&iga->iterator);CHKERRQ(ierr);

  iga->geometry = PETSC_FALSE;
  iga->rational = PETSC_FALSE;
  iga->vec_geom = PETSC_NULL;

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
  if (--((PetscObject)iga)->refct > 0) PetscFunctionReturn(0);

  ierr = PetscFree(iga->userops);CHKERRQ(ierr);
  ierr = PetscFree(iga->vectype);CHKERRQ(ierr);
  ierr = PetscFree(iga->mattype);CHKERRQ(ierr);
  if (iga->fieldname) {
    for (i=0; i<iga->dof; i++) {
      ierr = PetscFree(iga->fieldname[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(iga->fieldname);CHKERRQ(ierr);
  }

  for (i=0; i<3; i++) {
    ierr = IGAAxisDestroy(&iga->axis[i]);CHKERRQ(ierr);
    ierr = IGARuleDestroy(&iga->rule[i]);CHKERRQ(ierr);
    ierr = IGABasisDestroy(&iga->basis[i]);CHKERRQ(ierr);
    ierr = IGABoundaryDestroy(&iga->boundary[i][0]);CHKERRQ(ierr);
    ierr = IGABoundaryDestroy(&iga->boundary[i][1]);CHKERRQ(ierr);
  }
  ierr = IGAElementDestroy(&iga->iterator);CHKERRQ(ierr);

  ierr = IGAReset(iga);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(&iga);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAReset"
PetscErrorCode IGAReset(IGA iga)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!iga) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  iga->setup = PETSC_FALSE;
  ierr = IGAElementReset(iga->iterator);CHKERRQ(ierr);

  ierr = DMDestroy(&iga->dm_elem);CHKERRQ(ierr);

  iga->geometry = PETSC_FALSE;
  iga->rational = PETSC_FALSE;
  ierr = PetscFree(iga->geometryX);CHKERRQ(ierr);
  ierr = PetscFree(iga->geometryW);CHKERRQ(ierr);
  ierr = VecDestroy(&iga->vec_geom);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->dm_geom);CHKERRQ(ierr);
  
  ierr = AODestroy(&iga->ao);CHKERRQ(ierr);
  ierr = AODestroy(&iga->aob);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&iga->lgmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&iga->lgmapb);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->g2l);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->l2g);CHKERRQ(ierr);
  while (iga->nwork > 0)
    {ierr = VecDestroy(&iga->vwork[--iga->nwork]);CHKERRQ(ierr);}
  ierr = DMDestroy(&iga->dm_dof);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAView"
PetscErrorCode IGAView(IGA iga,PetscViewer viewer)
{
  PetscBool         isstring;
  PetscBool         isascii;
  PetscBool         isbinary;
  PetscViewerFormat format;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(((PetscObject)iga)->comm,&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(iga,1,viewer,2);
  if (!iga->setup) PetscFunctionReturn(0); /* XXX */
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (isbinary) {ierr = IGASave(iga,viewer);CHKERRQ(ierr);}
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII, &isascii );CHKERRQ(ierr);
  if (!isascii) PetscFunctionReturn(0);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  {
    MPI_Comm  comm;
    PetscBool geometry = iga->geometry ? PETSC_TRUE : PETSC_FALSE;
    PetscBool rational = iga->rational ? PETSC_TRUE : PETSC_FALSE;
    PetscInt  i,dim,dof;
    ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"IGA: dimension=%D  dofs/node=%D  geometry=%s  rational=%s\n",
                                  dim,dof,geometry?"yes":"no",rational?"yes":"no");CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      IGAAxis *AX = iga->axis;
      IGARule *QR = iga->rule;
      ierr = PetscViewerASCIIPrintf(viewer,"Axis %D: periodic=%d  degree=%D  quadrature=%D  processors=%D  nodes=%D  elements=%D\n",
                                    i,(int)AX[i]->periodic,AX[i]->p,QR[i]->nqp,
                                    iga->proc_sizes[i],iga->node_sizes[i],iga->elem_sizes[i]);CHKERRQ(ierr);
    }
    { /* */
      PetscInt isum[2],imin[2],imax[2],iloc[2] = {1, 1};
      for (i=0; i<dim; i++) {iloc[0] *= iga->node_width[i]; iloc[1] *= iga->elem_width[i];}
      ierr = MPI_Allreduce(&iloc,&isum,2,MPIU_INT,MPIU_SUM,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&iloc,&imin,2,MPIU_INT,MPIU_MIN,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&iloc,&imax,2,MPIU_INT,MPIU_MAX,comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Partitioning - nodes:    sum=%D  min=%D  max=%D  max/min=%g\n",
                                    isum[0],imin[0],imax[0],(double)imax[0]/(double)imin[0]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Partitioning - elements: sum=%D  min=%D  max=%D  max/min=%g\n",
                                    isum[1],imin[1],imax[1],(double)imax[1]/(double)imin[1]);CHKERRQ(ierr);
    }
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
        PetscMPIInt rank; PetscInt *ranks = iga->proc_rank;
        PetscInt *nnp = iga->node_width, tnnp = 1, *snp = iga->node_start;
        PetscInt *nel = iga->elem_width, tnel = 1, *sel = iga->elem_start;
        for (i=0; i<dim; i++) {tnnp *= nnp[i]; tnel *= nel[i];}
        ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] (%D,%D,%D): ",
                                                  (int)rank,ranks[0],ranks[1],ranks[2]);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"nodes=[%D:%D|%D:%D|%D:%D]=[%D|%D|%D]=%D  ",
                                                  snp[0],snp[0]+nnp[0]-1,
                                                  snp[1],snp[1]+nnp[1]-1,
                                                  snp[2],snp[2]+nnp[2]-1,
                                                  nnp[0],nnp[1],nnp[2],tnnp);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"elements=[%D:%D|%D:%D|%D:%D]=[%D|%D|%D]=%D\n",
                                                  sel[0],sel[0]+nel[0]-1,
                                                  sel[1],sel[1]+nel[1]-1,
                                                  sel[2],sel[2]+nel[2]-1,
                                                  nel[0],nel[1],nel[2],tnel);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
      }
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
  if (dim < 1 || dim > 3)
    SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
             "Number of parametric dimensions must be in range [1,3], got %D",dim);
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
#define __FUNCT__ "IGASetSpatialDim"
PetscErrorCode IGASetSpatialDim(IGA iga,PetscInt nsd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,nsd,2);
  if (nsd < 1 || nsd > 3)
    SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
             "Number of space dimensions must be in range [1,3], got %D",nsd);
  if (iga->nsd > 0 && iga->nsd != nsd)
    SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
             "Cannot change IGA dim from %D after it was set to %D",iga->nsd,nsd);
  iga->nsd = nsd;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetSpatialDim"
PetscErrorCode IGAGetSpatialDim(IGA iga,PetscInt *nsd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(nsd,2);
  *nsd = iga->nsd;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetDof"
PetscErrorCode IGASetDof(IGA iga,PetscInt dof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,dof,2);
  if (dof < 1)
    SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
             "Number of DOFs per node must be greather than one, got %D",dof);
  if (iga->dof > 0 && iga->dof != dof)
    SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
             "Cannot change number of DOFs from %D after it was set to %D",iga->dof,dof);
  iga->dof = dof;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetFieldName"
PetscErrorCode IGASetFieldName(IGA iga,PetscInt field,const char name[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidCharPointer(name,3);
  if (iga->dof < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
            "Must call IGASetDof() first");
  if (field < 0 || field >= iga->dof)
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Field number must be in range [0,%D], got %D",iga->dof-1,field);
  if (!iga->fieldname) {
    ierr = PetscMalloc1(iga->dof,char,&iga->fieldname);CHKERRQ(ierr);
    ierr = PetscMemzero(iga->fieldname,iga->dof*sizeof(char));CHKERRQ(ierr);
  }
  ierr = PetscFree(iga->fieldname[field]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&iga->fieldname[field]);CHKERRQ(ierr);
  if (iga->dm_dof) {ierr = DMDASetFieldName(iga->dm_dof,field,name);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetFieldName"
PetscErrorCode IGAGetFieldName(IGA iga,PetscInt field,const char *name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(name,3);
  if (iga->dof < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
            "Must call IGASetDof() first");
  if (field < 0 || field >= iga->dof)
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Field number must be in range [0,%D], got %D",iga->dof-1,field);
  if (iga->fieldname)
    *name = iga->fieldname[field];
  else
    *name = 0;
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
#define __FUNCT__ "IGAGetBasis"
PetscErrorCode IGAGetBasis(IGA iga,PetscInt i,IGABasis *basis)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(basis,3);
  if (iga->dim <= 0) SETERRQ (((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetDim() first");
  if (i < 0)         SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D must be nonnegative",i);
  if (i >= iga->dim) SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D, but dim %D",i,iga->dim);
  *basis = iga->basis[i];
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
  if (i < 0)         SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D must be nonnegative",i);
  if (i >= iga->dim) SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D, but dimension %D",i,iga->dim);
  if (iga->dof <= 0) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetDof() first");
  if (side < 0) side = 0; /* XXX error ?*/
  if (side > 1) side = 1; /* XXX error ?*/
  if (iga->boundary[i][side]->dof != iga->dof) {
    ierr = IGABoundaryInit(iga->boundary[i][side],iga->dof);CHKERRQ(ierr);
  }
  *boundary = iga->boundary[i][side];
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetOptionsPrefix"
PetscErrorCode IGAGetOptionsPrefix(IGA iga,const char *prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)iga,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetOptionsPrefix"
PetscErrorCode IGASetOptionsPrefix(IGA iga,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)iga,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPrependOptionsPrefix"
PetscErrorCode IGAPrependOptionsPrefix(IGA iga,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  ierr = PetscObjectPrependOptionsPrefix((PetscObject)iga,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAppendOptionsPrefix"
PetscErrorCode IGAAppendOptionsPrefix(IGA iga,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)iga,prefix);CHKERRQ(ierr);
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
    PetscInt  i,n;
    PetscBool perds[3]    = {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE};
    PetscInt  np,procs[3] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
    PetscInt  no,degrs[3] = {2,2,2};
    PetscInt  nc,conts[3] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
    PetscInt  ne,elems[3] = {16,16,16}; /* XXX Too coarse for 1D/2D ? */
    PetscReal bbox[3][2]  = {{0,1},{0,1},{0,1}};

    char vtype[256] = VECSTANDARD;
    char mtype[256] = MATBAIJ;
    PetscInt dim = iga->dim;
    PetscInt dof = iga->dof;

    for (i=0; i<dim; i++) perds[i] = iga->axis[i]->periodic;
    for (i=0; i<dim; i++) degrs[i] = iga->axis[i]->p;

    ierr = PetscObjectOptionsBegin((PetscObject)iga);CHKERRQ(ierr);
    if (iga->setup) goto setupcalled;
    ierr = PetscOptionsInt("-iga_dim","Number of dimensions",   "IGASetDim",iga->dim,&dim,&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetDim(iga,dim);CHKERRQ(ierr);}
    ierr = PetscOptionsInt("-iga_dof","Number of DOFs per node","IGASetDof",iga->dof,&dof,&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetDof(iga,dof);CHKERRQ(ierr);}
    if (iga->dim < 1) dim = 3;
    /* */
    ierr = PetscOptionsIntArray("-iga_processors","Processor grid","IGASetProcessors",procs,(np=dim,&np),&flg);CHKERRQ(ierr);
    if (flg) for (i=0; i<np; i++) {
        iga->proc_sizes[i] = procs[i]; /* XXX Use IGGASetProcessors() */
      }
    ierr = PetscOptionsBoolArray("-iga_periodic","Periodicity","IGAAxisSetPeriodic",perds,(n=dim,&n),&flg);CHKERRQ(ierr);
    if (flg) for (i=0; i<dim; i++) {
        PetscBool periodic = (i<n) ? perds[i] : perds[0];
        ierr = IGAAxisSetPeriodic(iga->axis[i],periodic);CHKERRQ(ierr);
      }
    ierr = PetscOptionsIntArray("-iga_degree","Polynomial degree","IGAAxisSetDegree",degrs,(no=dim,&no),&flg);CHKERRQ(ierr);
    if (flg) for (i=0; i<dim; i++) {
        PetscInt degree = (i<no) ? degrs[i] : degrs[0];
        ierr = IGAAxisSetDegree(iga->axis[i],degree);CHKERRQ(ierr);
      }
    ierr = PetscOptionsRealArray("-iga_bounding_box", "Bounding box", "IGAAxisInitUniform",&bbox[0][0],(n=2*dim,&n),&flg);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-iga_continuity","Continuity","IGAAxisInitUniform",conts,(nc=dim,&nc),&flg);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-iga_elements","Elements","IGAAxisInitUniform",elems,(ne=dim,&ne),&flg);CHKERRQ(ierr);
    if (flg) for (i=0; i<dim; i++) {
        PetscInt continuity = (i<nc) ? conts[i] : conts[0];
        PetscInt elements   = (i<ne) ? elems[i] : elems[0];
        PetscReal *U        = (i<n) ? &bbox[i][0] : &bbox[0][0];
        if (iga->axis[i]->p < 1) {ierr = IGAAxisSetDegree(iga->axis[i],degrs[i]);CHKERRQ(ierr);} /* XXX Default degree? */
        ierr = IGAAxisInitUniform(iga->axis[i],elements,U[0],U[1],continuity);CHKERRQ(ierr);
      }
  setupcalled:
    /* */
    if (iga->dof == 1) {ierr = PetscStrcpy(mtype,MATAIJ);CHKERRQ(ierr);}
    if (iga->vectype)  {ierr = PetscStrncpy(vtype,iga->vectype,sizeof(vtype));CHKERRQ(ierr);}
    if (iga->mattype)  {ierr = PetscStrncpy(mtype,iga->mattype,sizeof(mtype));CHKERRQ(ierr);}
    ierr = PetscOptionsList("-iga_vec_type","Vector type","VecSetType",VecList,vtype,vtype,sizeof vtype,&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetVecType(iga,vtype);CHKERRQ(ierr);}
    ierr = PetscOptionsList("-iga_mat_type","Matrix type","MatSetType",MatList,mtype,mtype,sizeof mtype,&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetMatType(iga,mtype);CHKERRQ(ierr);}
    /* */
    ierr = PetscOptionsName("-iga_view",         "Information on IGA context",       "IGAView",PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-iga_view_info",    "Output more detailed information", "IGAView",PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-iga_view_detailed","Output more detailed information", "IGAView",PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-iga_view_binary",  "Save to file in binary format",    "IGAView",PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)iga);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateSubComms1D"
PetscErrorCode IGACreateSubComms1D(IGA iga,MPI_Comm subcomms[])
{
  MPI_Comm       comm;
  PetscInt       i,dim;
  PetscMPIInt    size;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(subcomms,2);
  IGACheckSetUp(iga,1);

  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1 || dim == 1)
    for (i=0; i<dim; i++)
      {ierr = MPI_Comm_dup(comm,&subcomms[i]);CHKERRQ(ierr);}
  else {
    MPI_Comm    cartcomm;
    PetscMPIInt i,ndims,dims[3],periods[3]={0,0,0},reorder=0;
    ndims = PetscMPIIntCast(dim);
    for (i=0; i<ndims; i++) dims[i] = (PetscInt)iga->proc_sizes[ndims-1-i];
    ierr = MPI_Cart_create(comm,ndims,dims,periods,reorder,&cartcomm);CHKERRQ(ierr);
    for (i=0; i<ndims; i++) {
      PetscMPIInt remain_dims[3] = {0,0,0};
      remain_dims[ndims-1-i] = 1;
      ierr = MPI_Cart_sub(cartcomm,remain_dims,&subcomms[i]);CHKERRQ(ierr);
    }
    ierr = MPI_Comm_free(&cartcomm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateElemDM"
PetscErrorCode IGACreateElemDM(IGA iga,PetscInt bs,DM *dm_elem)
{
  DM             dm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  PetscValidPointer(dm_elem,3);
  IGACheckSetUp(iga,1);
  {
    PetscInt         i,dim;
    PetscInt         procs[3]   = {-1,-1,-1};
    PetscInt         sizes[3]   = { 1, 1, 1};
    PetscInt         *ranges[3] = { 0, 0, 0};
    PetscInt         swidth     = 0;
    DMDABoundaryType btype[3]   = {DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE};
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      sizes[i] = iga->axis[i]->nel;
      if (iga->proc_sizes[i] > 0)
        procs[i] = iga->proc_sizes[i];
    }
    ierr = DMDACreate(((PetscObject)iga)->comm,&dm);CHKERRQ(ierr);
    ierr = DMDASetDim(dm,dim);CHKERRQ(ierr);
    ierr = DMDASetDof(dm,bs);CHKERRQ(ierr);
    ierr = DMDASetNumProcs(dm,procs[0],procs[1],procs[2]);CHKERRQ(ierr);
    ierr = DMDASetSizes(dm,sizes[0],sizes[1],sizes[2]); CHKERRQ(ierr);
    ierr = DMDASetOwnershipRanges(dm,ranges[0],ranges[1],ranges[2]);CHKERRQ(ierr);
    ierr = DMDASetStencilType(dm,DMDA_STENCIL_BOX); CHKERRQ(ierr);
    ierr = DMDASetStencilWidth(dm,swidth); CHKERRQ(ierr);
    ierr = DMDASetBoundaryType(dm,btype[0],btype[1],btype[2]); CHKERRQ(ierr);
    ierr = DMSetUp(dm);CHKERRQ(ierr);
  }
  *dm_elem = dm;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateGeomDM"
PetscErrorCode IGACreateGeomDM(IGA iga,PetscInt bs,DM *dm_geom)
{
  DM             dm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  PetscValidPointer(dm_geom,3);
  IGACheckSetUp(iga,1);
  {
    PetscInt         i,dim;
    MPI_Comm         subcomms[3];
    PetscInt         procs[3]   = {-1,-1,-1};
    PetscInt         sizes[3]   = { 1, 1, 1};
    PetscInt         width[3]   = { 1, 1, 1};
    PetscInt         *ranges[3] = { 0, 0, 0};
    PetscInt         swidth     = 0;
    DMDABoundaryType btype[3]   = {DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE};
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    ierr = IGACreateSubComms1D(iga,subcomms);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      procs[i] = iga->proc_sizes[i];
      sizes[i] = iga->geom_sizes[i];
      width[i] = iga->geom_lwidth[i];
      ierr = PetscMalloc1(procs[i],PetscInt,&ranges[i]);CHKERRQ(ierr);
      ierr = MPI_Allgather(&width[i],1,MPIU_INT,ranges[i],1,MPIU_INT,subcomms[i]);CHKERRQ(ierr);
    }
    ierr = DMDACreate(((PetscObject)iga)->comm,&dm);CHKERRQ(ierr);
    ierr = DMDASetDim(dm,dim);CHKERRQ(ierr);
    ierr = DMDASetDof(dm,bs);CHKERRQ(ierr);
    ierr = DMDASetNumProcs(dm,procs[0],procs[1],procs[2]);CHKERRQ(ierr);
    ierr = DMDASetSizes(dm,sizes[0],sizes[1],sizes[2]); CHKERRQ(ierr);
    ierr = DMDASetOwnershipRanges(dm,ranges[0],ranges[1],ranges[2]);CHKERRQ(ierr);
    ierr = DMDASetStencilType(dm,DMDA_STENCIL_BOX); CHKERRQ(ierr);
    ierr = DMDASetStencilWidth(dm,swidth); CHKERRQ(ierr);
    ierr = DMDASetBoundaryType(dm,btype[0],btype[1],btype[2]); CHKERRQ(ierr);
    ierr = DMSetUp(dm);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      ierr = PetscFree(ranges[i]);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&subcomms[i]);CHKERRQ(ierr);
    }
  }
  *dm_geom = dm;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateNodeDM"
PetscErrorCode IGACreateNodeDM(IGA iga,PetscInt bs,DM *dm_node)
{
  DM             dm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  PetscValidPointer(dm_node,3);
  IGACheckSetUp(iga,1);
  {
    PetscInt         i,dim;
    MPI_Comm         subcomms[3];
    PetscInt         procs[3]   = {-1,-1,-1};
    PetscInt         sizes[3]   = { 1, 1, 1};
    PetscInt         width[3]   = { 1, 1, 1};
    PetscInt         *ranges[3] = { 0, 0, 0};
    PetscInt         swidth     = 0;
    DMDABoundaryType btype[3]   = {DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE};
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    ierr = IGACreateSubComms1D(iga,subcomms);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      procs[i] = iga->proc_sizes[i];
      sizes[i] = iga->node_sizes[i];
      width[i] = iga->node_width[i];
      ierr = PetscMalloc1(procs[i],PetscInt,&ranges[i]);CHKERRQ(ierr);
      ierr = MPI_Allgather(&width[i],1,MPIU_INT,ranges[i],1,MPIU_INT,subcomms[i]);CHKERRQ(ierr);
    }
    ierr = DMDACreate(((PetscObject)iga)->comm,&dm);CHKERRQ(ierr);
    ierr = DMDASetDim(dm,dim);CHKERRQ(ierr);
    ierr = DMDASetDof(dm,bs);CHKERRQ(ierr);
    ierr = DMDASetNumProcs(dm,procs[0],procs[1],procs[2]);CHKERRQ(ierr);
    ierr = DMDASetSizes(dm,sizes[0],sizes[1],sizes[2]); CHKERRQ(ierr);
    ierr = DMDASetOwnershipRanges(dm,ranges[0],ranges[1],ranges[2]);CHKERRQ(ierr);
    ierr = DMDASetStencilType(dm,DMDA_STENCIL_BOX); CHKERRQ(ierr);
    ierr = DMDASetStencilWidth(dm,swidth); CHKERRQ(ierr);
    ierr = DMDASetBoundaryType(dm,btype[0],btype[1],btype[2]); CHKERRQ(ierr);
    ierr = DMSetUp(dm);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      ierr = PetscFree(ranges[i]);CHKERRQ(ierr);
      ierr = MPI_Comm_free(&subcomms[i]);CHKERRQ(ierr);
    }
  }
  *dm_node = dm;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_CreateAO"
PetscErrorCode IGA_Grid_CreateAO(MPI_Comm comm,
                                 PetscInt dim,PetscInt bs,
                                 const PetscInt grid_sizes[],
                                 const PetscInt local_start[],
                                 const PetscInt local_width[],
                                 AO *ao)
{
  PetscInt       i;
  PetscInt       sizes[3]  = {1,1,1};
  PetscInt       lstart[3] = {0,0,0};
  PetscInt       lwidth[3] = {1,1,1};
  PetscInt       napp,*iapp;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidIntPointer(grid_sizes,4);
  PetscValidIntPointer(local_start,5);
  PetscValidIntPointer(local_width,6);
  PetscValidPointer(ao,7);
  for (i=0; i<dim; i++) {
    sizes[i]  = grid_sizes[i];
    lstart[i] = local_start[i];
    lwidth[i] = local_width[i];
  }
  {
    /* global grid strides */
    PetscInt jstride = sizes[0];
    PetscInt kstride = sizes[0]*sizes[1];
    /* local non-ghosted grid */
    PetscInt ilstart = lstart[0], ilend = lstart[0]+lwidth[0];
    PetscInt jlstart = lstart[1], jlend = lstart[1]+lwidth[1];
    PetscInt klstart = lstart[2], klend = lstart[2]+lwidth[2];
    PetscInt c,i,j,k,pos = 0;
    napp = lwidth[0]*lwidth[1]*lwidth[2];
    ierr = PetscMalloc1(napp*bs,PetscInt,&iapp);CHKERRQ(ierr);
    for (k=klstart; k<klend; k++)
      for (j=jlstart; j<jlend; j++)
        for (i=ilstart; i<ilend; i++)
          for (c=0; c<bs; c++)
            iapp[pos++] = (i + j * jstride + k * kstride)*bs + c;
  }
  ierr = AOCreateMemoryScalable(comm,napp,iapp,PETSC_NULL,ao);CHKERRQ(ierr);
  ierr = PetscFree(iapp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_CreateLGMap"
PetscErrorCode IGA_Grid_CreateLGMap(MPI_Comm comm,
                                    PetscInt dim,PetscInt bs,
                                    const PetscInt grid_sizes[],
                                    const PetscInt ghost_start[],
                                    const PetscInt ghost_width[],
                                    AO ao,LGMap *lgmap)
{
  PetscInt       i;
  PetscInt       sizes[3]  = {1,1,1};
  PetscInt       gstart[3] = {0,0,0};
  PetscInt       gwidth[3] = {1,1,1};
  PetscInt       nghost,*ighost;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidIntPointer(grid_sizes,4);
  PetscValidIntPointer(ghost_start,5);
  PetscValidIntPointer(ghost_width,6);
  PetscValidHeaderSpecific(ao,AO_CLASSID,7);
  PetscValidPointer(lgmap,8);
  for (i=0; i<dim; i++) {
    sizes[i]  = grid_sizes[i];
    gstart[i] = ghost_start[i];
    gwidth[i] = ghost_width[i];
  }
  {
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
    nghost = gwidth[0]*gwidth[1]*gwidth[2];
    ierr = PetscMalloc1(nghost*bs,PetscInt,&ighost);CHKERRQ(ierr);
    for (k=kgstart; k<kgend; k++) {
      for (j=jgstart; j<jgend; j++) {
        for (i=igstart; i<igend; i++) {
          PetscInt ig = i, jg = j, kg = k; /* account for periodicicty */
          if (ig<0) ig = isize + ig; else if (ig>=isize) ig = ig % isize;
          if (jg<0) jg = jsize + jg; else if (jg>=jsize) jg = jg % jsize;
          if (kg<0) kg = ksize + kg; else if (kg>=ksize) kg = kg % ksize;
          for (c=0; c<bs; c++)
            ighost[pos++] = (ig + jg * jstride + kg * kstride)*bs + c;
        }
      }
    }
  }
  /* map indices in global natural numbering to global petsc numbering */
  ierr = AOApplicationToPetsc(ao,nghost,ighost);CHKERRQ(ierr);
  /* create the local to global mapping */
  ierr = ISLocalToGlobalMappingCreate(comm,nghost,ighost,PETSC_OWN_POINTER,lgmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_CreateVector"
PetscErrorCode IGA_Grid_CreateVector(MPI_Comm comm,
                                     PetscInt dim,PetscInt bs,
                                     const PetscInt grid_sizes[],
                                     const PetscInt local_width[],
                                     const PetscInt ghost_width[],
                                     const VecType vectype,
                                     Vec *gvec, Vec *lvec)
{
  PetscInt       i;
  PetscInt       sizes[3]  = {1,1,1};
  PetscInt       lwidth[3] = {1,1,1};
  PetscInt       gwidth[3] = {1,1,1};
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidIntPointer(grid_sizes,4);
  PetscValidIntPointer(local_width,5);
  PetscValidIntPointer(ghost_width,6);
  if (vectype) PetscValidCharPointer(vectype,7);
  if (gvec) PetscValidPointer(gvec,8);
  if (lvec) PetscValidPointer(lvec,9);
  for (i=0; i<dim; i++) {
    sizes[i]  = grid_sizes[i];
    lwidth[i] = local_width[i];
    gwidth[i] = ghost_width[i];
  }
  if (gvec) {
    PetscInt n = lwidth[0]*lwidth[1]*lwidth[2];
    PetscInt N = sizes[0]*sizes[1]*sizes[2];
    ierr = VecCreate(comm,gvec);CHKERRQ(ierr);
    ierr = VecSetSizes(*gvec,n*bs,N*bs);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*gvec,bs);CHKERRQ(ierr);
    if (vectype) {ierr = VecSetType(*gvec,vectype);CHKERRQ(ierr);}
  }
  if (lvec) {
    PetscInt n = gwidth[0]*gwidth[1]*gwidth[2];
    ierr = VecCreate(PETSC_COMM_SELF,lvec);CHKERRQ(ierr);
    ierr = VecSetSizes(*lvec,n*bs,n*bs);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*lvec,bs);CHKERRQ(ierr);
    if (vectype) {ierr = VecSetType(*lvec,vectype);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_Grid_CreateScatter"
PetscErrorCode IGA_Grid_CreateScatter(MPI_Comm comm,
                                      PetscInt dim,PetscInt bs,
                                      const PetscInt local_start[],const PetscInt local_width[],
                                      const PetscInt ghost_start[],const PetscInt ghost_width[],
                                      LGMap lgmap,Vec gvec,Vec lvec,
                                      VecScatter *g2l,VecScatter *l2g)
{
  PetscInt       i;
  PetscInt       lstart[3] = {0,0,0};
  PetscInt       lwidth[3] = {1,1,1};
  PetscInt       gstart[3] = {0,0,0};
  PetscInt       gwidth[3] = {1,1,1};
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidIntPointer(local_start,4);
  PetscValidIntPointer(local_width,5);
  PetscValidIntPointer(ghost_start,6);
  PetscValidIntPointer(ghost_width,7);
  if (g2l) PetscValidHeaderSpecific(lgmap,IS_LTOGM_CLASSID,8);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,9);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,10);
  if (g2l) PetscValidPointer(g2l,11);
  if (l2g) PetscValidPointer(l2g,12);
  for (i=0; i<dim; i++) {
    lstart[i] = local_start[i];
    lwidth[i] = local_width[i];
    gstart[i] = ghost_start[i];
    gwidth[i] = ghost_width[i];
  }

  if (g2l) { /* build the global to local ghosted  scatter */
    IS isghost;
    PetscInt nghost;
    const PetscInt *ighost;
    ierr = ISLocalToGlobalMappingGetSize(lgmap,&nghost);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(lgmap,&ighost);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,nghost,ighost,PETSC_USE_POINTER,&isghost);CHKERRQ(ierr);
    ierr = VecScatterCreate(gvec,isghost,lvec,PETSC_NULL,g2l);CHKERRQ(ierr);
    ierr = ISDestroy(&isghost);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreIndices(lgmap,&ighost);CHKERRQ(ierr);
  }

  if (l2g) { /* build the local non-ghosted to global scatter */
    /* local non-ghosted grid */
    PetscInt ilstart = lstart[0], ilend = lstart[0]+lwidth[0];
    PetscInt jlstart = lstart[1], jlend = lstart[1]+lwidth[1];
    PetscInt klstart = lstart[2], klend = lstart[2]+lwidth[2];
    /* local ghosted grid */
    PetscInt igstart = gstart[0], igend = gstart[0]+gwidth[0];
    PetscInt jgstart = gstart[1], jgend = gstart[1]+gwidth[1];
    PetscInt kgstart = gstart[2], kgend = gstart[2]+gwidth[2];
    IS isglobal,islocal;
    PetscInt start,nlocal,*ilocal;
    PetscInt c,i,j,k,pos = 0,index = 0;
    ierr = VecGetLocalSize(gvec,&nlocal);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(gvec,&start,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(nlocal,PetscInt,&ilocal);CHKERRQ(ierr);
    for (k=kgstart; k<kgend; k++)
      for (j=jgstart; j<jgend; j++)
        for (i=igstart; i<igend; i++, index++)
          if (i>=ilstart && i<ilend && j>=jlstart && j<jlend && k>=klstart && k<klend)
            for (c=0; c<bs; c++) ilocal[pos++] = index*bs + c;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,nlocal,ilocal,PETSC_OWN_POINTER,&islocal);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,nlocal,start,1,&isglobal);CHKERRQ(ierr);
    ierr = VecScatterCreate(lvec,islocal,gvec,isglobal,l2g);CHKERRQ(ierr);
    ierr = ISDestroy(&islocal);CHKERRQ(ierr);
    ierr = ISDestroy(&isglobal);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateAO"
PetscErrorCode IGACreateAO(IGA iga,PetscInt bs,AO *ao)
{
  MPI_Comm       comm;
  const PetscInt *sizes,*lstart,*lwidth;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  PetscValidPointer(ao,3);
  IGACheckSetUp(iga,1);

  comm   = ((PetscObject)iga)->comm;
  sizes  = iga->node_sizes;
  lstart = iga->node_start;
  lwidth = iga->node_width;
  ierr = IGA_Grid_CreateAO(comm,iga->dim,bs,sizes,lstart,lwidth,ao);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateLGMap"
PetscErrorCode IGACreateLGMap(IGA iga,PetscInt bs,LGMap *lgmap)
{
  MPI_Comm       comm;
  PetscInt       *sizes,*gstart,*gwidth;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  PetscValidPointer(lgmap,3);
  IGACheckSetUp(iga,1);

  comm   = ((PetscObject)iga)->comm;
  sizes  = iga->node_sizes;
  gstart = iga->ghost_start;
  gwidth = iga->ghost_width;
  ierr = IGA_Grid_CreateLGMap(comm,iga->dim,1,sizes,gstart,gwidth,iga->aob,lgmap);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateVector"
PetscErrorCode IGACreateVector(IGA iga,PetscInt bs,Vec *global,Vec *local)
{
  MPI_Comm       comm;
  PetscInt       *sizes,*lwidth,*gwidth;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  if (global) PetscValidPointer(global,3);
  if (local)  PetscValidPointer(local,4);
  IGACheckSetUp(iga,1);

  comm   = ((PetscObject)iga)->comm;
  sizes  = iga->node_sizes;
  lwidth = iga->node_width;
  gwidth = iga->ghost_width;
  ierr = IGA_Grid_CreateVector(comm,iga->dim,bs,sizes,lwidth,gwidth,iga->vectype,global,local);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateScatter"
PetscErrorCode IGACreateScatter(IGA iga,PetscInt bs,Vec *gvec,Vec *lvec,VecScatter *g2l,VecScatter *l2g)
{
  MPI_Comm       comm;
  PetscInt       *lstart,*lwidth;
  PetscInt       *gstart,*gwidth;
  Vec            vglobal;
  Vec            vghost;
  LGMap          lgmap;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  if (gvec) PetscValidPointer(gvec,3);
  if (lvec) PetscValidPointer(lvec,4);
  if (g2l)  PetscValidPointer(g2l,5);
  if (l2g)  PetscValidPointer(l2g,6);
  IGACheckSetUp(iga,1);
  /* get the matching local to global mapping */
  if (bs == iga->dof) {
    lgmap = iga->lgmap;
    ierr = PetscObjectReference((PetscObject)lgmap);CHKERRQ(ierr);
  } else {
    ierr = ISLocalToGlobalMappingUnBlock(iga->lgmapb,bs,&lgmap);CHKERRQ(ierr);
  }
  /* create global and local ghosted vectors */
  ierr = IGACreateVector(iga,bs,&vglobal,&vghost);CHKERRQ(ierr);
  if (gvec) *gvec = vglobal;
  if (lvec) *lvec = vghost;

  comm   = ((PetscObject)iga)->comm;
  lstart = iga->node_start;
  lwidth = iga->node_width;
  gstart = iga->ghost_start;
  gwidth = iga->ghost_width;
  ierr = IGA_Grid_CreateScatter(comm,iga->dim,bs,lstart,lwidth,gstart,gwidth,
                                 lgmap,vglobal,vghost,g2l,l2g);CHKERRQ(ierr);

  if (!gvec) {ierr = VecDestroy(&vglobal);CHKERRQ(ierr);}
  if (!lvec) {ierr = VecDestroy(&vghost );CHKERRQ(ierr);}
  ierr = ISLocalToGlobalMappingDestroy(&lgmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetUp"
PetscErrorCode IGASetUp(IGA iga)
{
  PetscInt       i;
  PetscInt       p_max;
  DM             dm_elem;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (iga->setup) PetscFunctionReturn(0);

  if (iga->dim < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
            "Must call IGASetDim() first");

  if (iga->nsd < 1) /* XXX */
    iga->nsd = iga->dim;

  if (iga->dof < 1)
    iga->dof = 1;  /* XXX Error ? */

  for (i=0; i<iga->dim; i++) {
    ierr = IGAAxisSetUp(iga->axis[i]);CHKERRQ(ierr);
  }
  for (i=iga->dim; i<3; i++) {
    ierr = IGAAxisReset(iga->axis[i]);CHKERRQ(ierr);
    ierr = IGARuleReset(iga->rule[i]);CHKERRQ(ierr);
    ierr = IGABasisReset(iga->basis[i]);CHKERRQ(ierr);
    ierr = IGABoundaryReset(iga->boundary[i][0]);CHKERRQ(ierr);
    ierr = IGABoundaryReset(iga->boundary[i][1]);CHKERRQ(ierr);
  }

  p_max = 0;
  for (i=0; i<iga->dim; i++) {
    PetscInt p = iga->axis[i]->p;
    p_max = PetscMax(p_max,p);
  }
  for (i=0; i<3; i++) {
    PetscInt p = iga->axis[i]->p;
    PetscInt q = p+1; /* XXX */
    PetscInt d = PetscMin(p_max,3); /* XXX */
    ierr = IGARuleInit(iga->rule[i],q);CHKERRQ(ierr);
    ierr = IGABasisInit(iga->basis[i],iga->axis[i],iga->rule[i],d);CHKERRQ(ierr);
  }

  if (!iga->vectype) {
    const MatType vtype = VECSTANDARD;
    ierr = PetscStrallocpy(vtype,&iga->vectype);CHKERRQ(ierr);
  }
  if (!iga->mattype) {
    const MatType mtype = (iga->dof > 1) ? MATBAIJ : MATAIJ;
    ierr = PetscStrallocpy(mtype,&iga->mattype);CHKERRQ(ierr);
  }

  iga->setup = PETSC_TRUE;

  ierr = IGACreateElemDM(iga,1,&dm_elem);CHKERRQ(ierr);
  { /* processor grid and coordinates */
    MPI_Comm    comm = ((PetscObject)iga)->comm;
    PetscInt    *proc_rank  = iga->proc_rank;
    PetscInt    *proc_sizes = iga->proc_sizes;
    PetscMPIInt index;
    ierr = DMDAGetInfo(dm_elem,0,0,0,0,
                       &proc_sizes[0],&proc_sizes[1],&proc_sizes[2],
                       0,0,0,0,0,0);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&index);CHKERRQ(ierr);
    for (i=0; i<iga->dim; i++) {
      proc_rank[i] = index % proc_sizes[i];
      index -= proc_rank[i];
      index /= proc_sizes[i];
    }
    for (i=iga->dim; i<3; i++) {
      proc_rank[i]  = 0;
      proc_sizes[i] = 1;
    }
  }
  { /* element partitioning */
    PetscInt *elem_sizes = iga->elem_sizes;
    PetscInt *elem_start = iga->elem_start;
    PetscInt *elem_width = iga->elem_width;
    ierr = DMDAGetInfo(dm_elem,0,
                       &elem_sizes[0],&elem_sizes[1],&elem_sizes[2],
                       0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
    ierr = DMDAGetCorners(dm_elem,
                          &elem_start[0],&elem_start[1],&elem_start[2],
                          &elem_width[0],&elem_width[1],&elem_width[2]);CHKERRQ(ierr);
    for (i=iga->dim; i<3; i++) {
      elem_sizes[i] = 1;
      elem_start[i] = 0;
      elem_width[i] = 1;
    }
  }
  ierr = DMDestroy(&dm_elem);CHKERRQ(ierr);
  { /* node partitioning */
    IGAAxis  *AX = iga->axis;
    PetscInt *elem_start  = iga->elem_start;
    PetscInt *elem_width  = iga->elem_width;
    PetscInt *node_sizes  = iga->node_sizes;
    PetscInt *node_start  = iga->node_start;
    PetscInt *node_width  = iga->node_width;
    PetscInt *ghost_start = iga->ghost_start;
    PetscInt *ghost_width = iga->ghost_width;
    for (i=0; i<iga->dim; i++) {
      PetscBool wrap = AX[i]->periodic;
      PetscInt nel = AX[i]->nel;
      PetscInt nnp = AX[i]->nnp;
      PetscInt p = AX[i]->p;
      PetscInt *span = AX[i]->span;
      PetscInt efirst = elem_start[i];
      PetscInt elast  = elem_start[i] + elem_width[i] - 1;
      PetscInt nfirst = 0, nlast = nnp - 1;
      PetscInt mid = wrap ? 0 : p/2; /* XXX Is this optimal? */
      if (efirst > 0     ) nfirst = span[efirst-1] - p + mid + 1;
      if (elast  < nel-1 ) nlast  = span[elast]    - p + mid;
      node_sizes[i]  = nnp;
      node_start[i]  = nfirst;
      node_width[i]  = nlast + 1 - nfirst;
      ghost_start[i] = span[efirst] - p;
      ghost_width[i] = span[elast]  + p + 1 - span[efirst];
    }
    for (i=iga->dim; i<3; i++) {
      node_sizes[i]  = 1;
      node_start[i]  = 0;
      node_width[i]  = 1;
      ghost_start[i] = 0;
      ghost_width[i] = 1;
    }
  }
  { /* geometry partitioning */
    PetscInt *geom_sizes  = iga->geom_sizes;
    PetscInt *geom_lstart = iga->geom_lstart;
    PetscInt *geom_lwidth = iga->geom_lwidth;
    PetscInt *geom_gstart = iga->geom_gstart;
    PetscInt *geom_gwidth = iga->geom_gwidth;
    for (i=0; i<iga->dim; i++) {
      PetscInt rank = iga->proc_rank[i];
      PetscInt size = iga->proc_sizes[i];
      PetscInt m = iga->axis[i]->m;
      PetscInt p = iga->axis[i]->p;
      PetscInt n = m - p - 1;
      geom_sizes[i]  = n + 1;
      geom_lstart[i] = iga->node_start[i];
      geom_lwidth[i] = iga->node_width[i];
      geom_gstart[i] = iga->ghost_start[i];
      geom_gwidth[i] = iga->ghost_width[i];
      if (rank == size-1)
        geom_lwidth[i] = geom_sizes[i] - geom_gstart[i];
    }
    for (i=iga->dim; i<3; i++) {
      geom_sizes[i]  = 1;
      geom_lstart[i] = 0;
      geom_lwidth[i] = 1;
      geom_gstart[i] = 0;
      geom_gwidth[i] = 1;
    }
  }

  /* */
  ierr = IGACreateNodeDM(iga,iga->dof,&iga->dm_dof);CHKERRQ(ierr);
  if (iga->fieldname)
    for (i=0; i<iga->dof; i++)
      {ierr = DMDASetFieldName(iga->dm_dof,i,iga->fieldname[i]);CHKERRQ(ierr);}
  /* build the block application ordering */
  ierr = IGACreateAO(iga,1,&iga->aob);CHKERRQ(ierr);
  /* build the scalar and block local to global mappings */
  ierr = IGACreateLGMap(iga,iga->dof,&iga->lgmapb);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingUnBlock(iga->lgmapb,iga->dof,&iga->lgmap);CHKERRQ(ierr);
  /* build global to local and local to global vector scatters */
  ierr = IGACreateScatter(iga,iga->dof,PETSC_NULL,PETSC_NULL,&iga->g2l,&iga->l2g);CHKERRQ(ierr);

  iga->iterator->parent = iga;
  ierr = IGAElementSetUp(iga->iterator);CHKERRQ(ierr);


  { /* */
    PetscBool flg1,flg2,info=PETSC_FALSE;
    char filename1[PETSC_MAX_PATH_LEN] = "";
    char filename2[PETSC_MAX_PATH_LEN] = "";
    PetscViewer viewer;
    ierr = PetscObjectOptionsBegin((PetscObject)iga);CHKERRQ(ierr);
    ierr = PetscOptionsString("-iga_view",         "Information on IGA context",       "IGAView",filename1,filename1,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
    ierr = PetscOptionsBool(  "-iga_view_info",    "Output more detailed information", "IGAView",info,&info,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool(  "-iga_view_detailed","Output more detailed information", "IGAView",info,&info,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-iga_view_binary",  "Save to file in binary format",    "IGAView",filename2,filename2,PETSC_MAX_PATH_LEN,&flg2);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if ((flg1||info) && !PetscPreLoadingOn) {
      ierr = PetscViewerASCIIOpen(((PetscObject)iga)->comm,filename1,&viewer);CHKERRQ(ierr);
      if (info) {ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);}
      ierr = IGAView(iga,viewer);CHKERRQ(ierr);
      if (info) {ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);}
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    if (flg2 && !PetscPreLoadingOn) {
      PetscViewer newviewer=0;
      if (filename2[0]) {
        ierr = PetscViewerBinaryOpen(((PetscObject)iga)->comm,filename2,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
        newviewer = viewer;
      } else {
        viewer = PETSC_VIEWER_BINARY_(((PetscObject)iga)->comm);
        PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
      }
      ierr = IGAView(iga,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&newviewer);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetVecType"
PetscErrorCode IGASetVecType(IGA iga,const VecType vectype)
{
  VecType        vtype;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidCharPointer(vectype,2);
  ierr = PetscStrallocpy(vectype,&vtype);CHKERRQ(ierr);
  ierr = PetscFree(iga->vectype);CHKERRQ(ierr);
  iga->vectype = vtype;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetMatType"
PetscErrorCode IGASetMatType(IGA iga,const MatType mattype)
{
  MatType        mtype;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidCharPointer(mattype,2);
  ierr = PetscStrallocpy(mattype,&mtype);CHKERRQ(ierr);
  ierr = PetscFree(iga->mattype);CHKERRQ(ierr);
  iga->mattype = mtype;
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

#undef  __FUNCT__
#define __FUNCT__ "IGASetUserIEFunction"
PetscErrorCode IGASetUserIEFunction(IGA iga,IGAUserIEFunction IEFunction,void *IEFunCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (IEFunction) iga->userops->IEFunction = IEFunction;
  if (IEFunCtx)   iga->userops->IEFunCtx   = IEFunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetUserIEJacobian"
PetscErrorCode IGASetUserIEJacobian(IGA iga,IGAUserIEJacobian IEJacobian,void *IEJacCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (IEJacobian) iga->userops->IEJacobian = IEJacobian;
  if (IEJacCtx)   iga->userops->IEJacCtx   = IEJacCtx;
  PetscFunctionReturn(0);
}

#if PETSC_VERSION_(3,2,0)
#include "private/dmimpl.h"
#undef  __FUNCT__
#define __FUNCT__ "DMSetMatType"
PetscErrorCode DMSetMatType(DM dm,const MatType mattype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = PetscFree(dm->mattype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(mattype,&dm->mattype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
