#include "petiga.h"
#include "petigagrid.h"

#undef  __FUNCT__
#define __FUNCT__ "IGACreate"
/*@
   IGACreate - Creates the default IGA context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  _iga - location to put the IGA context

   Level: normal

.keywords: IGA, create
@*/
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
    ierr = IGAAxisCreate(&iga->axis[i]);CHKERRQ(ierr);
    ierr = IGARuleCreate(&iga->rule[i]);CHKERRQ(ierr);
    ierr = IGABasisCreate(&iga->basis[i]);CHKERRQ(ierr);
    ierr = IGABoundaryCreate(&iga->boundary[i][0]);CHKERRQ(ierr);
    ierr = IGABoundaryCreate(&iga->boundary[i][1]);CHKERRQ(ierr);
    iga->proc_sizes[i] = -1;
  }
  ierr = IGAElementCreate(&iga->iterator);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGADestroy"
/*@
   IGADestroy - Destroys the IGA context.

   Collective on IGA

   Input Parameter:
.  _iga - context obtained from IGACreate

   Level: normal

.keywords: IGA, destroy
@*/
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

  /* element */
  ierr = DMDestroy(&iga->dm_elem);CHKERRQ(ierr);
  /* geometry */
  iga->geometry = PETSC_FALSE;
  iga->rational = PETSC_FALSE;
  ierr = PetscFree(iga->geometryX);CHKERRQ(ierr);
  ierr = PetscFree(iga->geometryW);CHKERRQ(ierr);
  ierr = VecDestroy(&iga->vec_geom);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->dm_geom);CHKERRQ(ierr);
  /* node */
  ierr = DMDestroy(&iga->dm_node);CHKERRQ(ierr);
  ierr = AODestroy(&iga->ao);CHKERRQ(ierr);
  ierr = AODestroy(&iga->aob);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&iga->lgmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&iga->lgmapb);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->g2l);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->l2g);CHKERRQ(ierr);
  while (iga->nwork > 0)
    {ierr = VecDestroy(&iga->vwork[--iga->nwork]);CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAView"
PetscErrorCode IGAView(IGA iga,PetscViewer viewer)
{
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

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (isbinary) { ierr = IGASave(iga,viewer);CHKERRQ(ierr); PetscFunctionReturn(0); }

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII, &isascii );CHKERRQ(ierr);
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
      for (i=0; i<dim; i++) {iloc[0] *= iga->node_lwidth[i]; iloc[1] *= iga->elem_width[i];}
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
        PetscInt *nnp = iga->node_lwidth, tnnp = 1, *snp = iga->node_lstart;
        PetscInt *nel = iga->elem_width,  tnel = 1, *sel = iga->elem_start;
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
/*@
   IGASetDim - Sets the dimension of the parameter space

   Logically Collective on IGA

   Input Parameters:
+  iga - the IGA context
-  dim - the dimension of the parameter space

   Level: normal

.keywords: IGA, dimension
@*/
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
/*@
   IGASetSpatialDim - Sets the dimension of the geometry

   Logically Collective on IGA

   Input Parameters:
+  iga - the IGA context
-  dim - the dimension of the geometry

   Level: normal

.keywords: IGA, dimension
@*/
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
/*@
   IGASetDof - Sets the number of degrees of freedom per basis

   Logically Collective on IGA

   Input Parameters:
+  iga - the IGA context
-  dof - the number of dofs per basis

   Level: normal

.keywords: IGA, dofs
@*/
PetscErrorCode IGASetDof(IGA iga,PetscInt dof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,dof,2);
  if (dof < 1)
    SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
             "Number of DOFs per node must be greater than one, got %D",dof);
  if (iga->dof > 0 && iga->dof != dof)
    SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
             "Cannot change number of DOFs from %D after it was set to %D",iga->dof,dof);
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
#define __FUNCT__ "IGASetFieldName"
/*@
   IGASetFieldName - Sets the names of individual field components in
   multicomponent vectors associated with a IGA.

   Not Collective

   Input Parameters:
+  iga - the IGA context
.  field - the field number associated to a dof of the IGA (0,1,...dof-1)
-  name - the name of the field

   Level: normal

.keywords: IGA, field, name
@*/
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
    ierr = PetscMalloc1(iga->dof,char*,&iga->fieldname);CHKERRQ(ierr);
    ierr = PetscMemzero(iga->fieldname,iga->dof*sizeof(char*));CHKERRQ(ierr);
  }
  ierr = PetscFree(iga->fieldname[field]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&iga->fieldname[field]);CHKERRQ(ierr);
  if (iga->dm_node) {ierr = DMDASetFieldName(iga->dm_node,field,name);CHKERRQ(ierr);}
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
#define __FUNCT__ "IGASetProcessors"
PetscErrorCode IGASetProcessors(IGA iga,PetscInt i,PetscInt processors)
{
  PetscMPIInt    size;
  PetscInt       k,dim,np[3],prod;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  dim = (iga->dim > 0) ? iga->dim : 3;
  if (i <    0) SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Axis index must be nonnegative, got %D",i);
  if (i >= dim) SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Axis index must be in range [0,%D], got %D",dim-1,i);
  ierr = MPI_Comm_size(((PetscObject)iga)->comm,&size);CHKERRQ(ierr);
  if (processors < 1)
    SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of processors must be nonnegative, got %D",processors);
  if (size % processors != 0)
    SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of processors %D is incompatible with communicator size %d",processors,(int)size);
  for (k=0; k<dim; k++)
    np[k] = iga->proc_sizes[k];
  np[i] = prod = processors;
  for (k=0; k<dim; k++)
    if (k!=i && np[k]>0) prod *= np[k];
  if (size % prod != 0)
    SETERRQ4(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,
             "Processor grid sizes (%D,%D,%D) are incompatible with communicator size %d",np[0],np[1],np[2],(int)size);
  iga->proc_sizes[i] = processors;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetAxis"
/*@
   IGAGetAxis - Returns a pointer to the i^th parametric axis associated with the IGA

   Not Collective

   Input Parameters:
+  iga - the IGA context
-  i - the axis index

   Output Parameter:
.  axis - the axis context

   Level: normal

.keywords: IGA, axis
@*/
PetscErrorCode IGAGetAxis(IGA iga,PetscInt i,IGAAxis *axis)
{
  PetscInt dim;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(axis,3);
  dim = (iga->dim > 0) ? iga->dim : 3;
  if (i <    0) SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Axis index must be nonnegative, got %D",i);
  if (i >= dim) SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Axis index must be in range [0,%D], got %D",dim-1,i);
  *axis = iga->axis[i];
  PetscFunctionReturn(0);
}

/*@
   IGAGetRule - Returns a pointer to the i^th quadrature rule associated with the IGA

   Not Collective

   Input Parameters:
+  iga - the IGA context
-  i - the axis index

   Output Parameter:
.  rule - the quadrature rule context

   Level: normal

.keywords: IGA, quadrature rule
@*/
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
/*@
   IGAGetBoundary - Returns a pointer to a specific side of the i^th
   boundary associated with the IGA.

   Not Collective

   Input Parameters:
+  iga - the IGA context
.  i - the boundary index
-  side - the side index: 0 (left) or 1 (right)

   Output Parameter:
.  boundary - the boundary context

   Notes:
   A side marker of 0 corresponds to the boundary associated to the
   minimum knot value of the i^th axis. A side marker of 1 corresponds
   to the boundary associated to the maximum knot value of the i^th
   axis.

   Level: normal

.keywords: IGA, boundary
@*/
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
#define __FUNCT__ "IGAGetBasis"
PetscErrorCode IGAGetBasis(IGA iga,PetscInt i,IGABasis *basis)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(basis,3);
  IGACheckSetUp(iga,1);
  if (i < 0)         SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D must be nonnegative",i);
  if (i >= iga->dim) SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D, but dim %D",i,iga->dim);
  *basis = iga->basis[i];
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
/*@
   IGASetFromOptions - Call this in your code to allow IGA options to
   be set from the command line. This routine should be called before
   IGASetUp().

   Collective on IGA

   Input Parameter:
.  iga - the IGA context

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetFromOptions(IGA iga)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  {
    PetscBool flg;
    PetscInt  i,nw,nb;
    PetscBool wraps[3]    = {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE};
    PetscInt  np,procs[3] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
    PetscInt  nd,degrs[3] = {2,2,2};
    PetscInt  nq,quadr[3] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
    PetscInt  nc,conts[3] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
    PetscInt  ne,elems[3] = {16,16,16};
    PetscReal bbox[3][2]  = {{0,1},{0,1},{0,1}};
    char      filename[PETSC_MAX_PATH_LEN] = {0};
    char      vtype[256] = VECSTANDARD;
    char      mtype[256] = MATBAIJ;
    PetscInt  dim = iga->dim;

    /* Periodicity, degree, and quadrature are initially what they are intially set to */
    for (i=0; i<dim; i++) wraps[i] = iga->axis[i]->periodic;
    for (i=0; i<dim; i++) if (iga->axis[i]->p   > 0) degrs[i] = iga->axis[i]->p;
    for (i=0; i<dim; i++) if (iga->rule[i]->nqp > 0) quadr[i] = iga->rule[i]->nqp;

    ierr = PetscObjectOptionsBegin((PetscObject)iga);CHKERRQ(ierr);

    /* If setup has been called, then many options are not available so skip them. */
    if (iga->setup) goto setupcalled;

    /* Processor grid */
    ierr = PetscOptionsIntArray("-iga_processors","Processor grid","IGASetProcessors",procs,(np=dim,&np),&flg);CHKERRQ(ierr);
    if (flg) for (i=0; i<np; i++) {
        PetscInt np = procs[i];
        if (np > 0) {ierr = IGASetProcessors(iga,i,np);CHKERRQ(ierr);}
      }

    /* Periodicity */
    ierr = PetscOptionsBoolArray("-iga_periodic","Periodicity","IGAAxisSetPeriodic",wraps,(nw=dim,&nw),&flg);CHKERRQ(ierr);
    if (flg) for (i=0; i<dim; i++) {
        PetscBool w = (i<nw) ? wraps[i] : wraps[0];
        if (nw == 0) w = PETSC_TRUE;
        ierr = IGAAxisSetPeriodic(iga->axis[i],w);CHKERRQ(ierr);
      }

    /* Geometry */
    ierr = PetscOptionsString   ("-iga_geometry","Specify IGA geometry file","IGARead",filename,filename,sizeof(filename),&flg);CHKERRQ(ierr);
    if (flg) { /* load from file */
      ierr = IGARead(iga,filename);CHKERRQ(ierr);
      ierr = PetscOptionsReject("-iga_degree",    PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReject("-iga_limits",    PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReject("-iga_continuity",PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReject("-iga_elements",  PETSC_NULL);CHKERRQ(ierr);
    } else { /* set axis details */
      ierr = PetscOptionsIntArray ("-iga_degree",    "Degree",    "IGAAxisSetDegree",  degrs,(nd=dim,&nd),PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsRealArray("-iga_limits",    "Limits",    "IGAAxisInitUniform",&bbox[0][0],(nb=2*dim,&nb),PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsIntArray ("-iga_continuity","Continuity","IGAAxisInitUniform",conts,(nc=dim,&nc),PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsIntArray ("-iga_elements",  "Elements",  "IGAAxisInitUniform",elems,(ne=dim,&ne),PETSC_NULL);CHKERRQ(ierr);
      for (i=0; i<dim; i++) {
        PetscBool  w = iga->axis[i]->periodic;
        PetscInt   p = (i<nd) ? degrs[i] : degrs[0];
        PetscInt   C = (i<nc) ? conts[i] : conts[0];
        PetscInt   N = (i<ne) ? elems[i] : elems[0];
        PetscReal *U = (i<nb/2) ? &bbox[i][0] : &bbox[0][0];
        if (p < 1) p = iga->axis[i]->p; if (p < 1) p = 2;
        ierr = IGAAxisReset(iga->axis[i]);CHKERRQ(ierr);
        ierr = IGAAxisSetPeriodic(iga->axis[i],w);CHKERRQ(ierr);
        ierr = IGAAxisSetDegree(iga->axis[i],p);CHKERRQ(ierr);
        ierr = IGAAxisInitUniform(iga->axis[i],N,U[0],U[1],C);CHKERRQ(ierr);
      }
    }

    /* Quadrature rule */
    ierr = PetscOptionsIntArray ("-iga_quadrature","Quadrature points","IGARuleInit",quadr,(nq=dim,&nq),&flg);CHKERRQ(ierr);
    if (flg) for (i=0; i<dim; i++) {
        PetscInt q = (i<nq) ? quadr[i] : quadr[0];
        if (q > 0) {ierr = IGARuleInit(iga->rule[i],q);CHKERRQ(ierr);}
      }

  setupcalled:
    /* Matrix and Vector type */
    if (iga->dof == 1) {ierr = PetscStrcpy(mtype,MATAIJ);CHKERRQ(ierr);}
    if (iga->vectype)  {ierr = PetscStrncpy(vtype,iga->vectype,sizeof(vtype));CHKERRQ(ierr);}
    if (iga->mattype)  {ierr = PetscStrncpy(mtype,iga->mattype,sizeof(mtype));CHKERRQ(ierr);}
    ierr = PetscOptionsList("-iga_vec_type","Vector type","IGASetVecType",VecList,vtype,vtype,sizeof vtype,&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetVecType(iga,vtype);CHKERRQ(ierr);}
    ierr = PetscOptionsList("-iga_mat_type","Matrix type","IGASetMatType",MatList,mtype,mtype,sizeof mtype,&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetMatType(iga,mtype);CHKERRQ(ierr);}
    /* View options, handled in IGASetUp() */
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
/*@
   IGACreateNodeDM - Creates a DM using the distributed pattern of the
   nodes of the IGA.

   Collective on IGA

   Input Parameters:
+  iga - the IGA context
-  bs - the block size (number of degrees of freedom)

   Output Parameter:
.  dm_node - the DM

   Notes:
   We have built PetIGA in such a way that interaction with
   coefficients of the solution are rarely needed. On occasion this is
   needed, as for example when setting nonzero initial conditions for
   a transient problem. This routine can be used to address these
   needs.

   Level: normal

.keywords: IGA, create DM, access dof grid
@*/
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
      width[i] = iga->node_lwidth[i];
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
#define __FUNCT__ "IGASetUp"
/*@
   IGASetUp - Sets up the internal data structures for the later use of the IGA.

   Collective on IGA

   Input Parameter:
.  iga - the IGA context

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetUp(IGA iga)
{
  PetscInt       i;
  DM             dm_elem;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (iga->setup) PetscFunctionReturn(0);

  if (iga->dim < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
            "Must call IGASetDim() first");

  if (iga->dof < 1)
    iga->dof = 1;  /* XXX Error ? */

  if (iga->nsd < 1) /* XXX */
    iga->nsd = iga->dim;

  for (i=0; i<iga->dim; i++) {
    ierr = IGAAxisSetUp(iga->axis[i]);CHKERRQ(ierr);
    if (iga->rule[i]->nqp < 1) {
      PetscInt p = iga->axis[i]->p;
      PetscInt q = p + 1;
      ierr = IGARuleInit(iga->rule[i],q);CHKERRQ(ierr);
    }
  }

  for (i=iga->dim; i<3; i++) {
    ierr = IGAAxisReset(iga->axis[i]);CHKERRQ(ierr);
    ierr = IGARuleReset(iga->rule[i]);CHKERRQ(ierr);
    ierr = IGABoundaryReset(iga->boundary[i][0]);CHKERRQ(ierr);
    ierr = IGABoundaryReset(iga->boundary[i][1]);CHKERRQ(ierr);
  }

  { /* */
    PetscInt order = 0;
    for (i=0; i<iga->dim; i++) {
      PetscInt p = iga->axis[i]->p;
      order = PetscMax(order,p);
    }
    order = PetscMin(order,3); /* XXX */
    order = PetscMax(order,1); /* XXX */
    iga->order = order;
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
    PetscInt *node_lstart = iga->node_lstart;
    PetscInt *node_lwidth = iga->node_lwidth;
    PetscInt *node_gstart = iga->node_gstart;
    PetscInt *node_gwidth = iga->node_gwidth;
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
      node_lstart[i] = nfirst;
      node_lwidth[i] = nlast + 1 - nfirst;
      node_gstart[i] = span[efirst] - p;
      node_gwidth[i] = span[elast]  + p + 1 - span[efirst];
    }
    for (i=iga->dim; i<3; i++) {
      node_sizes[i]  = 1;
      node_lstart[i] = 0;
      node_lwidth[i] = 1;
      node_gstart[i] = 0;
      node_gwidth[i] = 1;
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
      geom_lstart[i] = iga->node_lstart[i];
      geom_lwidth[i] = iga->node_lwidth[i];
      geom_gstart[i] = iga->node_gstart[i];
      geom_gwidth[i] = iga->node_gwidth[i];
      if (rank == size-1)
        geom_lwidth[i] = geom_sizes[i] - geom_lstart[i];
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
  ierr = IGACreateNodeDM(iga,iga->dof,&iga->dm_node);CHKERRQ(ierr);
  if (iga->fieldname)
    for (i=0; i<iga->dof; i++)
      {ierr = DMDASetFieldName(iga->dm_node,i,iga->fieldname[i]);CHKERRQ(ierr);}
  {
    IGA_Grid grid;
    ierr = IGA_Grid_Create(((PetscObject)iga)->comm,&grid);CHKERRQ(ierr);
    ierr = IGA_Grid_Init(grid,
                         iga->dim,iga->dof,iga->node_sizes,
                         iga->node_lstart,iga->node_lwidth,
                         iga->node_gstart,iga->node_gwidth);CHKERRQ(ierr);

    /* build the block application ordering */
    ierr = IGA_Grid_GetAOBlock(grid,&iga->aob);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->aob);CHKERRQ(ierr);
    /* build the scalar and block local to global mappings */
    ierr = IGA_Grid_GetLGMap(grid,&iga->lgmap);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->lgmap);CHKERRQ(ierr);
    ierr = IGA_Grid_GetLGMapBlock(grid,&iga->lgmapb);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->lgmapb);CHKERRQ(ierr);
    /* build global to local and local to global vector scatters */
    ierr = IGA_Grid_GetScatterG2L(grid,&iga->g2l);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->g2l);CHKERRQ(ierr);
    ierr = IGA_Grid_GetScatterL2G(grid,&iga->l2g);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->l2g);CHKERRQ(ierr);

    ierr = IGA_Grid_Destroy(&grid);CHKERRQ(ierr);
  }

  for (i=0; i<3; i++) {
    ierr = IGABasisInit(iga->basis[i],iga->axis[i],iga->rule[i],iga->order);CHKERRQ(ierr);
  }
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
/*@
   IGASetUserIFunction - Set the function which computes the residual
   R(u)=0 for use with implicit time stepping routines.

   Logically Collective on TS

   Input Parameter:
+  iga - the IGA context
.  IFunction - the function evaluation routine
-  IFunCtx - user-defined context for private data for the function evaluation routine (may be PETSC_NULL)

   Details of IFunction:
$  PetscErrorCode IFunction(IGAPoint p,PetscReal dt,
                            PetscReal shift,const PetscScalar *V,
                            PetscReal t,const PetscScalar *U,
                            PetscScalar *R,void *ctx);

+  p - point at which to compute the residual
.  dt - time step size
.  shift - positive parameter which depends on the time integration method (XXX Should this be here?)
.  V - time derivative of the state vector
.  t - time at step/stage being solved
.  U - state vector
.  R - function vector
-  ctx - [optional] user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
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
/*@
   IGASetUserIJacobian - Set the function to compute the matrix dF/dU
   + shift*dF/dU_t where F(t,U,U_t) is the function you provided with
   IGASetUserIFunction().

   Logically Collective on TS

   Input Parameter:
+  iga - the IGA context
.  IJacobian - the Jacobian evaluation routine
-  IJacCtx - user-defined context for private data for the Jacobian evaluation routine (may be PETSC_NULL)

   Details of IJacobian:
$  PetscErrorCode IJacobian(IGAPoint p,PetscReal dt,
                            PetscReal shift,const PetscScalar *V,
                            PetscReal t,const PetscScalar *U,
                            PetscScalar *J,void *ctx);

+  p - point at which to compute the Jacobian
.  dt - time step size
.  shift - positive parameter which depends on the time integration method
.  V - time derivative of the state vector
.  t - time at step/stage being solved
.  U - state vector
.  J - Jacobian matrix
-  ctx - [optional] user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
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
/*@
   IGASetUserIEFunction - Set the function which computes the residual
   R(u)=0 for use with explicit or implicit time stepping routines.

   Logically Collective on TS

   Input Parameter:
+  iga - the IGA context
.  IEFunction - the function evaluation routine
-  IEFunCtx - user-defined context for private data for the function evaluation routine (may be PETSC_NULL)

   Details of IEFunction:
$  PetscErrorCode IEFunction(IGAPoint p,PetscReal dt,
                             PetscReal shift,const PetscScalar *V0,
                             PetscReal t1,const PetscScalar *U1,
                             PetscReal t0,const PetscScalar *U0,
                             PetscScalar *R,void *ctx);

+  p - point at which to compute the residual
.  dt - time step size
.  shift - positive parameter which depends on the time integration method (XXX Should this be here?)
.  V0 - time derivative of the state vector at t0
.  t1 - time at step/stage being solved
.  U1 - state vector at t1
.  t0 - time at current step
.  U0 - state vector at t0
.  R - function vector
-  ctx - [optional] user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
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
/*@
   IGASetUserIEJacobian - Set the function to compute the matrix dF/dU
   + shift*dF/dU_t where F(t,U,U_t) is the function you provided with
   IGASetUserIEFunction(). For use with implicit or explicit TS methods.

   Logically Collective on TS

   Input Parameter:
+  iga - the IGA context
.  IEJacobian - the Jacobian evaluation routine
-  IEJacCtx - user-defined context for private data for the Jacobian evaluation routine (may be PETSC_NULL)

   Details of IEJacobian:
$  PetscErrorCode IEJacobian(IGAPoint p,PetscReal dt,
                             PetscReal shift,const PetscScalar *V0,
                             PetscReal t1,const PetscScalar *U1,
                             PetscReal t0,const PetscScalar *U0,
                             PetscScalar *J,void *ctx);

+  p - point at which to compute the Jacobian
.  dt - time step size
.  shift - positive parameter which depends on the time integration method
.  V0 - time derivative of the state vector at t0
.  t1 - time at step/stage being solved
.  U1 - state vector at t1
.  t0 - time at current step
.  U0 - state vector at t0
.  J - Jacobian matrix
-  ctx - [optional] user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetUserIEJacobian(IGA iga,IGAUserIEJacobian IEJacobian,void *IEJacCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (IEJacobian) iga->userops->IEJacobian = IEJacobian;
  if (IEJacCtx)   iga->userops->IEJacCtx   = IEJacCtx;
  PetscFunctionReturn(0);
}
