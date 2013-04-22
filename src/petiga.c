#include "petiga.h"
#include "petigapart.h"
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
  ierr = IGAInitializePackage();CHKERRQ(ierr);
#endif
#if PETSC_VERSION_LE(3,3,0)
  ierr = PetscHeaderCreate(iga,_p_IGA,struct _IGAOps,IGA_CLASSID,-1,
                           "IGA","IGA","IGA",comm,IGADestroy,IGAView);CHKERRQ(ierr);
#else
  ierr = PetscHeaderCreate(iga,_p_IGA,struct _IGAOps,IGA_CLASSID,
                           "IGA","IGA","IGA",comm,IGADestroy,IGAView);CHKERRQ(ierr);
#endif

  *_iga = iga;

  ierr = PetscNew(struct _IGAUserOps,&iga->userops);CHKERRQ(ierr);
  iga->vectype = PETSC_NULL;
  iga->mattype = PETSC_NULL;

  iga->dim = -1;
  iga->dof = -1;
  iga->order = -1;

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

  for (i=0; i<3; i++) /* collocation */
    {ierr = IGABasisDestroy(&iga->node_basis[i]);CHKERRQ(ierr);}
  ierr = IGAElementDestroy(&iga->node_iterator);CHKERRQ(ierr);

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
  iga->setupstage = 0;

  /* element */
  ierr = DMDestroy(&iga->elem_dm);CHKERRQ(ierr);
  /* geometry */
  iga->geometry = 0;
  iga->rational = PETSC_FALSE;
  iga->property = 0;
  ierr = PetscFree(iga->geometryX);CHKERRQ(ierr);
  ierr = PetscFree(iga->rationalW);CHKERRQ(ierr);
  ierr = PetscFree(iga->propertyA);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->geom_dm);CHKERRQ(ierr);
  /* node */
  ierr = AODestroy(&iga->ao);CHKERRQ(ierr);
  ierr = AODestroy(&iga->aob);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&iga->lgmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&iga->lgmapb);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->g2l);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->l2g);CHKERRQ(ierr);
  ierr = VecDestroy(&iga->natural);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->n2g);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->g2n);CHKERRQ(ierr);
  while (iga->nwork > 0)
    {ierr = VecDestroy(&iga->vwork[--iga->nwork]);CHKERRQ(ierr);}
  ierr = DMDestroy(&iga->node_dm);CHKERRQ(ierr);

  ierr = IGAElementReset(iga->iterator);CHKERRQ(ierr);
  ierr = IGAElementReset(iga->node_iterator);CHKERRQ(ierr);

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
    MPI_Comm comm;
    PetscInt i,dim,dof,order;
    ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);
    ierr = IGAGetOrder(iga,&order);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"IGA: dim=%D  dof=%D  order=%D  geometry=%D  rational=%D  property=%D\n",
                                  dim,dof,order,iga->geometry,(PetscInt)iga->rational,iga->property);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"Axis %D: periodic=%d  degree=%D  quadrature=%D  processors=%D  nodes=%D  elements=%D\n",
                                    i,(int)iga->axis[i]->periodic,iga->axis[i]->p,iga->rule[i]->nqp,
                                    iga->proc_sizes[i],iga->node_sizes[i],iga->elem_sizes[i]);CHKERRQ(ierr);
    }
    { /* */
      PetscInt isum[2],imin[2],imax[2],iloc[2] = {1, 1};
      for (i=0; i<dim; i++) {iloc[0] *= iga->node_lwidth[i]; iloc[1] *= iga->elem_width[i];}
      ierr = MPI_Allreduce(iloc,isum,2,MPIU_INT,MPIU_SUM,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(iloc,imin,2,MPIU_INT,MPIU_MIN,comm);CHKERRQ(ierr);
      ierr = MPI_Allreduce(iloc,imax,2,MPIU_INT,MPIU_MAX,comm);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Partitioning - nodes:    sum=%D  min=%D  max=%D  max/min=%g\n",
                                    isum[0],imin[0],imax[0],(double)imax[0]/(double)imin[0]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Partitioning - elements: sum=%D  min=%D  max=%D  max/min=%g\n",
                                    isum[1],imin[1],imax[1],(double)imax[1]/(double)imin[1]);CHKERRQ(ierr);
    }
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
        PetscMPIInt rank; PetscInt *ranks = iga->proc_ranks;
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
  char           *fname;
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
    ierr = PetscMalloc1(iga->dof+1,char*,&iga->fieldname);CHKERRQ(ierr);
    ierr = PetscMemzero(iga->fieldname,(iga->dof+1)*sizeof(char*));CHKERRQ(ierr);
  }
  ierr = PetscStrallocpy(name,&fname);CHKERRQ(ierr);
  ierr = PetscFree(iga->fieldname[field]);CHKERRQ(ierr);
  iga->fieldname[field] = fname;
  if (iga->node_dm) {ierr = DMDASetFieldName(iga->node_dm,field,name);CHKERRQ(ierr);}
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
#define __FUNCT__ "IGASetOrder"
PetscErrorCode IGASetOrder(IGA iga,PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,order,2);
  if (order < 0)
    SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
             "Order must be nonnegative, got %D",order);
  if (iga->order >= 0 && iga->order != order)
    SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
             "Cannot change order from %D after it was set to %D",iga->order,order);
  iga->order = order;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetOrder"
PetscErrorCode IGAGetOrder(IGA iga,PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidIntPointer(order,2);
  *order= iga->order;
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
  if (iga->setup) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot call after IGASetUp()");
  if (i < 0)      SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D must be nonnegative",i);
  if (i >= dim)   SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D, but dim %D",i,dim);
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
#define __FUNCT__ "IGASetUseCollocation"
PetscErrorCode IGASetUseCollocation(IGA iga,PetscBool collocation)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveBool(iga,collocation,2);
  if (collocation && !iga->collocation) {
    PetscMPIInt size = 1;
    PetscInt i, dim = (iga->dim > 0) ? iga->dim : 3;
    PetscBool periodic = PETSC_FALSE;
    ierr = MPI_Comm_size(((PetscObject)iga)->comm,&size);CHKERRQ(ierr);
    for (i=0; i<dim; i++) if(iga->axis[i]->periodic) periodic = PETSC_TRUE;
    if (size > 1) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_SUP,
                          "Collocation not supported in parallel");
    if (periodic) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_SUP,
                          "Collocation not supported with periodicity");
  }
  if (collocation && iga->setup) { /* collocation */
    PetscInt i;
    for (i=0; i<3; i++) {
      ierr = IGABasisDestroy(&iga->node_basis[i]);CHKERRQ(ierr);
      ierr = IGABasisCreate(&iga->node_basis[i]);CHKERRQ(ierr);
      ierr = IGABasisInitCollocation(iga->node_basis[i],iga->axis[i],iga->order);CHKERRQ(ierr);
    }
    ierr = IGAElementDestroy(&iga->node_iterator);CHKERRQ(ierr);
    ierr = IGAElementCreate(&iga->node_iterator);CHKERRQ(ierr);
    iga->node_iterator->collocation = PETSC_TRUE;
    ierr = IGAElementInit(iga->node_iterator,iga);CHKERRQ(ierr);
  }
  iga->collocation = collocation;
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
  if (i < 0)    SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D must be nonnegative",i);
  if (i >= dim) SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D, but dim %D",i,dim);
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
  PetscInt dim;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(rule,3);
  dim = (iga->dim > 0) ? iga->dim : 3;
  if (i < 0)    SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D must be nonnegative",i);
  if (i >= dim) SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D, but dim %D",i,iga->dim);
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
  PetscInt       dim;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(boundary,4);
  dim = (iga->dim > 0) ? iga->dim : 3;
  if (i < 0)    SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D must be nonnegative",i);
  if (i >= dim) SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D, but dimension %D",i,iga->dim);
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
  if (i >= iga->dim) SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D, but dimension %D",i,iga->dim);
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
    PetscInt  i,nw,nl;
    IGABasisType btype[3] = {IGA_BASIS_BSPLINE,IGA_BASIS_BSPLINE,IGA_BASIS_BSPLINE};
    PetscBool    wraps[3] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE };
    PetscInt  np,procs[3] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
    PetscInt  nq,quadr[3] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
    PetscInt  ne,elems[3] = {16,16,16};
    PetscInt  nd,degrs[3] = { 2, 2, 2};
    PetscInt  nc,conts[3] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
    PetscReal ulims[3][2] = {{0,1},{0,1},{0,1}};
    char      filename[PETSC_MAX_PATH_LEN] = {0};
    char      vtype[256]  = VECSTANDARD;
    char      mtype[256]  = MATBAIJ;
    PetscInt  dim = (iga->dim > 0) ? iga->dim : 3;
    PetscInt  dof = (iga->dof > 0) ? iga->dof : 1;
    PetscInt  order = iga->order;

    for (i=0; i<dim; i++) {
      procs[i] = iga->proc_sizes[i];
      wraps[i] = iga->axis[i]->periodic;
      btype[i] = iga->basis[i]->type;
      if (iga->rule[i]->nqp > 0)
        quadr[i] = iga->rule[i]->nqp;
    }
    for (i=0; i<dim; i++) {
      IGAAxis axis = iga->axis[i];
      if (axis->p > 0)
        degrs[i] = axis->p;
      else if (i > 0)
        degrs[i] = degrs[i-1];
      if (axis->m > 1) {
        elems[i]    = axis->nel;
        ulims[i][0] = axis->U[axis->p];
        ulims[i][1] = axis->U[axis->m-axis->p];
      } else if (i > 0) {
        elems[i]    = elems[i-1];
        ulims[i][0] = ulims[i-1][0];
        ulims[i][1] = ulims[i-1][1];
      }
    }

    ierr = PetscObjectOptionsBegin((PetscObject)iga);CHKERRQ(ierr);

    /* If setup has been called, then many options are not available so skip them. */
    if (iga->setup) goto setupcalled;

    ierr = PetscOptionsInt("-iga_dim","Number of dimensions","IGASetDim",iga->dim,&dim,&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetDim(iga,dim);CHKERRQ(ierr);}
    dim = (iga->dim > 0) ? iga->dim : 3;

    ierr = PetscOptionsInt("-iga_dof","Number of DOFs per node","IGASetDof",iga->dof,&dof,&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetDof(iga,dof);CHKERRQ(ierr);}
    dof = (iga->dof > 0) ? iga->dof : 1;

    /* Processor grid */
    ierr = PetscOptionsIntArray("-iga_processors","Processor grid","IGASetProcessors",procs,(np=dim,&np),&flg);CHKERRQ(ierr);
    if (flg) for (i=0; i<np; i++) {
        PetscInt n = procs[i];
        if (n > 0) {ierr = IGASetProcessors(iga,i,n);CHKERRQ(ierr);}
      }

    /* Periodicity */
    ierr = PetscOptionsBoolArray("-iga_periodic","Periodicity","IGAAxisSetPeriodic",wraps,(nw=dim,&nw),&flg);CHKERRQ(ierr);
    if (flg && nw==0) for (i=0; i<dim; i++) wraps[i] = PETSC_TRUE;
    if (flg && nw==1) for (i=1; i<dim; i++) wraps[i] = wraps[0]; /* XXX */
    if (flg) for (i=0; i<dim; i++) {
        PetscBool w = wraps[i];
        ierr = IGAAxisSetPeriodic(iga->axis[i],w);CHKERRQ(ierr);
      }

    /* Basis */
    ierr = PetscOptionsEnum("-iga_basis_type","Basis type","IGABasisSetType",IGABasisTypes,(PetscEnum)btype[0],(PetscEnum*)&btype[0],&flg);CHKERRQ(ierr);
    for (i=0; i<dim; i++) btype[i] = btype[0]; /* XXX */
    if (flg) for (i=0; i<dim; i++) {
        ierr = IGABasisSetType(iga->basis[i],btype[i]);CHKERRQ(ierr);
      }
    for (i=0; i<dim; i++) if (btype[i] != IGA_BASIS_BSPLINE) conts[i] = 0;

    /* Geometry */
    ierr = PetscOptionsString("-iga_geometry","Specify IGA geometry file","IGARead",filename,filename,sizeof(filename),&flg);CHKERRQ(ierr);
    if (flg) { /* load from file */
      ierr = IGARead(iga,filename);CHKERRQ(ierr);
      ierr = PetscOptionsReject("-iga_elements",  PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReject("-iga_degree",    PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReject("-iga_continuity",PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsReject("-iga_limits",    PETSC_NULL);CHKERRQ(ierr);
    } else { /* set axis details */
      ierr = PetscOptionsIntArray ("-iga_elements",  "Elements",  "IGAAxisInitUniform",elems,(ne=dim,&ne),&flg);CHKERRQ(ierr);
      ierr = PetscOptionsIntArray ("-iga_degree",    "Degree",    "IGAAxisSetDegree",  degrs,(nd=dim,&nd),PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsIntArray ("-iga_continuity","Continuity","IGAAxisInitUniform",conts,(nc=dim,&nc),PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsRealArray("-iga_limits",    "Limits",    "IGAAxisInitUniform",&ulims[0][0],(nl=2*dim,&nl),PETSC_NULL);CHKERRQ(ierr);
      for (nl/=2, i=0; i<dim; i++) {
        IGAAxis axis = iga->axis[i];
        PetscBool  w = axis->periodic;
        PetscInt   p = (i<nd) ? degrs[i] : degrs[0];
        PetscInt   N = (i<ne) ? elems[i] : elems[0];
        PetscReal *L = (i<nl) ? ulims[i] : ulims[0];
        PetscInt   C = (i<nc) ? conts[i] : conts[0];
        if (p < 1) {if (axis->p > 0) p = axis->p;   else p =  2;}
        if (N < 1) {if (axis->m > 1) N = axis->nel; else N = 16;}
        if (flg || (axis->p==0||axis->m==1)) {
          ierr = IGAAxisReset(axis);CHKERRQ(ierr);
          ierr = IGAAxisSetPeriodic(axis,w);CHKERRQ(ierr);
          ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);
          ierr = IGAAxisInitUniform(axis,N,L[0],L[1],C);CHKERRQ(ierr);
        }
      }
    }

    /* Quadrature */
    for (i=0; i<dim; i++) if (quadr[i] < 1) quadr[i] = iga->axis[i]->p + 1;
    ierr = PetscOptionsIntArray("-iga_quadrature","Quadrature points","IGARuleInit",quadr,(nq=dim,&nq),&flg);CHKERRQ(ierr);
    if (flg) for (i=0; i<dim; i++) {
        PetscInt q = (i<nq) ? quadr[i] : quadr[0];
        if (q > 0) {ierr = IGARuleInit(iga->rule[i],q);CHKERRQ(ierr);}
      }

    /* Order */
    if (order < 0) for (i=0; i<dim; i++) order = PetscMax(order,iga->axis[i]->p);
    order = PetscMax(order,1); order = PetscMin(order,3);
    ierr = PetscOptionsInt("-iga_order","Order","IGASetOrder",order,&order,&flg);CHKERRQ(ierr);
    if (flg) { ierr = IGASetOrder(iga,order);CHKERRQ(ierr);}

  setupcalled:

    /* Collocation */ {
      PetscBool collocation = iga->collocation;
      ierr = PetscOptionsBool("-iga_collocation","Use collocation","IGASetCollocation",collocation,&collocation,&flg);CHKERRQ(ierr);
      if (flg) {ierr = IGASetUseCollocation(iga,collocation);CHKERRQ(ierr);}
    }

    /* Matrix and Vector type */
    if (iga->dof == 1) {ierr = PetscStrcpy(mtype,MATAIJ);CHKERRQ(ierr);}
    if (iga->vectype)  {ierr = PetscStrncpy(vtype,iga->vectype,sizeof(vtype));CHKERRQ(ierr);}
    if (iga->mattype)  {ierr = PetscStrncpy(mtype,iga->mattype,sizeof(mtype));CHKERRQ(ierr);}
    ierr = PetscOptionsList("-iga_vec_type","Vector type","IGASetVecType",VecList,vtype,vtype,sizeof(vtype),&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetVecType(iga,vtype);CHKERRQ(ierr);}
    ierr = PetscOptionsList("-iga_mat_type","Matrix type","IGASetMatType",MatList,mtype,mtype,sizeof(mtype),&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetMatType(iga,mtype);CHKERRQ(ierr);}

    /* View options, handled in IGASetUp() */
    ierr = PetscOptionsName("-iga_view",       "Information on IGA context",      "IGAView",PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-iga_view_info",  "Output more detailed information","IGAView",PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-iga_view_detail","Output more detailed information","IGAView",PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-iga_view_binary","Save to file in binary format",   "IGAView",PETSC_NULL);CHKERRQ(ierr);

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
#ifndef PETSC_HAVE_MPIUNI
  else {
    MPI_Comm    cartcomm;
    PetscMPIInt i,ndims,dims[3],periods[3]={0,0,0},reorder=0;
    ndims = (PetscMPIInt)dim;
    for (i=0; i<ndims; i++) dims[i] = (PetscMPIInt)iga->proc_sizes[ndims-1-i];
    ierr = MPI_Cart_create(comm,ndims,dims,periods,reorder,&cartcomm);CHKERRQ(ierr);
    for (i=0; i<ndims; i++) {
      PetscMPIInt remain_dims[3] = {0,0,0};
      remain_dims[ndims-1-i] = 1;
      ierr = MPI_Cart_sub(cartcomm,remain_dims,&subcomms[i]);CHKERRQ(ierr);
    }
    ierr = MPI_Comm_free(&cartcomm);CHKERRQ(ierr);
  }
#endif /* PETSC_HAVE_MPIUNI */
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateDM"
static
PetscErrorCode IGACreateDM(IGA iga,PetscInt bs,
                           const PetscInt gsizes[],
                           const PetscInt lsizes[],
                           const PetscBool periodic[],
                           PetscBool stencil_box,
                           PetscInt  stencil_width,
                           DM *dm_)
{
  PetscInt         i,dim;
  MPI_Comm         subcomms[3];
  PetscInt         procs[3]   = {-1,-1,-1};
  PetscInt         sizes[3]   = { 1, 1, 1};
  PetscInt         width[3]   = { 1, 1, 1};
  PetscInt         *ranges[3] = {PETSC_NULL, PETSC_NULL, PETSC_NULL};
  DMDABoundaryType btype[3]   = {DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE};
  DMDAStencilType  stype      = stencil_box ? DMDA_STENCIL_BOX : DMDA_STENCIL_STAR;
  PetscInt         swidth     = stencil_width;
  DM               dm;
  PetscErrorCode   ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  PetscValidIntPointer(gsizes,3);
  PetscValidIntPointer(lsizes,4);
  if (periodic) PetscValidIntPointer(periodic,5);
  PetscValidPointer(dm_,3);
  IGACheckSetUp(iga,1);

  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  for (i=0; i<dim; i++) {
    sizes[i] = gsizes[i];
    width[i] = lsizes[i];
    btype[i] = (periodic && periodic[i]) ? DMDA_BOUNDARY_PERIODIC : DMDA_BOUNDARY_NONE;
    procs[i] = iga->proc_sizes[i];
  }
  ierr = IGACreateSubComms1D(iga,subcomms);CHKERRQ(ierr);
  for (i=0; i<dim; i++) {
    ierr = PetscMalloc1(procs[i],PetscInt,&ranges[i]);CHKERRQ(ierr);
    ierr = MPI_Allgather(&width[i],1,MPIU_INT,ranges[i],1,MPIU_INT,subcomms[i]);CHKERRQ(ierr);
    ierr = MPI_Comm_free(&subcomms[i]);CHKERRQ(ierr);
  }
  ierr = DMDACreate(((PetscObject)iga)->comm,&dm);CHKERRQ(ierr);
  ierr = DMDASetDim(dm,dim);CHKERRQ(ierr);
  ierr = DMDASetDof(dm,bs);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(dm,procs[0],procs[1],procs[2]);CHKERRQ(ierr);
  ierr = DMDASetSizes(dm,sizes[0],sizes[1],sizes[2]); CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(dm,ranges[0],ranges[1],ranges[2]);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(dm,btype[0],btype[1],btype[2]);CHKERRQ(ierr);
  ierr = DMDASetStencilType(dm,stype);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(dm,swidth);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  for (i=0; i<dim; i++) {ierr = PetscFree(ranges[i]);CHKERRQ(ierr);}
  *dm_ = dm;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateElemDM"
PetscErrorCode IGACreateElemDM(IGA iga,PetscInt bs,DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  PetscValidPointer(dm,3);
  IGACheckSetUp(iga,1);
  ierr = IGACreateDM(iga,bs,iga->elem_sizes,iga->elem_width,
                     PETSC_NULL,PETSC_TRUE,0,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateGeomDM"
PetscErrorCode IGACreateGeomDM(IGA iga,PetscInt bs,DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  PetscValidPointer(dm,3);
  IGACheckSetUp(iga,1);
  ierr = IGACreateDM(iga,bs,iga->geom_sizes,iga->geom_lwidth,
                     PETSC_NULL,PETSC_TRUE,0,dm);CHKERRQ(ierr);
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
.  dm - the DM

   Notes:
   We have built PetIGA in such a way that interaction with
   coefficients of the solution are rarely needed. On occasion this is
   needed, as for example when setting nonzero initial conditions for
   a transient problem. This routine can be used to address these
   needs.

   Level: normal

.keywords: IGA, create DM, access dof grid
@*/
PetscErrorCode IGACreateNodeDM(IGA iga,PetscInt bs,DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  PetscValidPointer(dm,3);
  IGACheckSetUp(iga,1);
  ierr = IGACreateDM(iga,bs,iga->node_sizes,iga->node_lwidth,
                     PETSC_NULL,PETSC_TRUE,0,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetElemDM"
PetscErrorCode IGAGetElemDM(IGA iga,DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(dm,3);
  IGACheckSetUp(iga,1);
  if (!iga->elem_dm) {ierr = IGACreateElemDM(iga,iga->dof,&iga->elem_dm);CHKERRQ(ierr);}
  *dm = iga->elem_dm;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetGeomDM"
PetscErrorCode IGAGetGeomDM(IGA iga,DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(dm,3);
  IGACheckSetUp(iga,1);
  if (!iga->geom_dm) {ierr = IGACreateGeomDM(iga,iga->dof,&iga->geom_dm);CHKERRQ(ierr);}
  *dm = iga->geom_dm;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetNodeDM"
PetscErrorCode IGAGetNodeDM(IGA iga,DM *dm)
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(dm,3);
  IGACheckSetUp(iga,1);
  if (!iga->node_dm) {
    ierr = IGACreateNodeDM(iga,iga->dof,&iga->node_dm);CHKERRQ(ierr);
    if (iga->fieldname)
      for (i=0; i<iga->dof; i++)
        {ierr = DMDASetFieldName(iga->node_dm,i,iga->fieldname[i]);CHKERRQ(ierr);}
  }
  *dm = iga->node_dm;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetUp_Stage1"
PetscErrorCode IGASetUp_Stage1(IGA iga)
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (iga->setupstage >= 1) PetscFunctionReturn(0);
  iga->setupstage = 1;

  for (i=0; i<iga->dim; i++)
    {ierr = IGAAxisSetUp(iga->axis[i]);CHKERRQ(ierr);}
  for (i=iga->dim; i<3; i++)
    {ierr = IGAAxisReset(iga->axis[i]);CHKERRQ(ierr);}

  { /* processor grid and coordinates */
    MPI_Comm    comm = ((PetscObject)iga)->comm;
    PetscMPIInt size,rank;
    PetscInt    grid_sizes[3] = {1,1,1};
    PetscInt    *proc_sizes = iga->proc_sizes;
    PetscInt    *proc_ranks = iga->proc_ranks;
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    for (i=0; i<iga->dim; i++)
      grid_sizes[i] = iga->axis[i]->nel;
    ierr = IGA_Partition(size,rank,iga->dim,grid_sizes,
                         proc_sizes,proc_ranks);CHKERRQ(ierr);
    for (i=iga->dim; i<3; i++) {
      proc_sizes[i] = 1;
      proc_ranks[i] = 0;
    }
  }
  { /* element partitioning */
    PetscInt *elem_sizes = iga->elem_sizes;
    PetscInt *elem_start = iga->elem_start;
    PetscInt *elem_width = iga->elem_width;
    for (i=0; i<iga->dim; i++)
      elem_sizes[i] = iga->axis[i]->nel;
    ierr = IGA_Distribute(iga->dim,iga->proc_sizes,iga->proc_ranks,
                          elem_sizes,elem_width,elem_start);CHKERRQ(ierr);
    for (i=iga->dim; i<3; i++) {
      elem_sizes[i] = 1;
      elem_start[i] = 0;
      elem_width[i] = 1;
    }
  }
  { /* geometry partitioning */
    PetscInt *geom_sizes  = iga->geom_sizes;
    PetscInt *geom_lstart = iga->geom_lstart;
    PetscInt *geom_lwidth = iga->geom_lwidth;
    PetscInt *geom_gstart = iga->geom_gstart;
    PetscInt *geom_gwidth = iga->geom_gwidth;
    for (i=0; i<iga->dim; i++) {
      PetscInt nel    = iga->elem_sizes[i];
      PetscInt efirst = iga->elem_start[i];
      PetscInt elast  = iga->elem_start[i] + iga->elem_width[i] - 1;
      PetscInt p = iga->axis[i]->p;
      PetscInt *span = iga->axis[i]->span;
      PetscInt size,lstart,lend,gstart,gend;
      size = span[nel-1] + 1;
      gstart = span[efirst] - p;
      gend = span[elast] + 1;
      lstart = span[efirst] - p;
      if(elast < nel-1)
        lend = span[elast+1] - p;
      else
        lend = span[elast] + 1;
      geom_sizes[i]  = size;
      geom_lstart[i] = lstart;
      geom_lwidth[i] = lend - lstart;
      geom_gstart[i] = gstart;
      geom_gwidth[i] = gend - gstart;
    }
    for (i=iga->dim; i<3; i++) {
      geom_sizes[i]  = 1;
      geom_lstart[i] = 0;
      geom_lwidth[i] = 1;
      geom_gstart[i] = 0;
      geom_gwidth[i] = 1;
    }
  }
  /* element */
  ierr = DMDestroy(&iga->elem_dm);CHKERRQ(ierr);
  /* geometry */
  iga->geometry = 0;
  iga->rational = PETSC_FALSE;
  iga->property = 0;
  ierr = PetscFree(iga->geometryX);CHKERRQ(ierr);
  ierr = PetscFree(iga->rationalW);CHKERRQ(ierr);
  ierr = PetscFree(iga->propertyA);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->geom_dm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetUp_Stage2"
PetscErrorCode IGASetUp_Stage2(IGA iga)
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (iga->setupstage >= 2) PetscFunctionReturn(0);
  iga->setupstage = 2;

  { /* node partitioning */
    PetscInt *node_sizes  = iga->node_sizes;
    PetscInt *node_lstart = iga->node_lstart;
    PetscInt *node_lwidth = iga->node_lwidth;
    PetscInt *node_gstart = iga->node_gstart;
    PetscInt *node_gwidth = iga->node_gwidth;
    for (i=0; i<iga->dim; i++) {
      PetscInt size = iga->proc_sizes[i];
      PetscInt rank = iga->proc_ranks[i];
      node_sizes[i]  = iga->axis[i]->nnp; /* XXX */
      node_lstart[i] = iga->geom_lstart[i];
      node_lwidth[i] = iga->geom_lwidth[i];
      node_gstart[i] = iga->geom_gstart[i];
      node_gwidth[i] = iga->geom_gwidth[i];
      if (rank == size-1)
        node_lwidth[i] = node_sizes[i] - node_lstart[i];
    }
    for (i=iga->dim; i<3; i++) {
      node_sizes[i]  = 1;
      node_lstart[i] = 0;
      node_lwidth[i] = 1;
      node_gstart[i] = 0;
      node_gwidth[i] = 1;
    }
  }

  if (iga->dof < 1) iga->dof = 1;  /* XXX Error ? */

  ierr = AODestroy(&iga->ao);CHKERRQ(ierr);
  ierr = AODestroy(&iga->aob);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&iga->lgmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&iga->lgmapb);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->g2l);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->l2g);CHKERRQ(ierr);
  ierr = VecDestroy(&iga->natural);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->n2g);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->g2n);CHKERRQ(ierr);
  while (iga->nwork > 0)
    {ierr = VecDestroy(&iga->vwork[--iga->nwork]);CHKERRQ(ierr);}
  ierr = DMDestroy(&iga->node_dm);CHKERRQ(ierr);
  {
    IGA_Grid grid;
    /* create the grid context */
    ierr = IGA_Grid_Create(((PetscObject)iga)->comm,&grid);CHKERRQ(ierr);
    ierr = IGA_Grid_Init(grid,
                         iga->dim,iga->dof,iga->node_sizes,
                         iga->node_lstart,iga->node_lwidth,
                         iga->node_gstart,iga->node_gwidth);CHKERRQ(ierr);
    /* build the scalar and block application orderings */
    ierr = IGA_Grid_GetAO(grid,&iga->ao);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->ao);CHKERRQ(ierr);
    ierr = IGA_Grid_GetAOBlock(grid,&iga->aob);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->aob);CHKERRQ(ierr);
    /* build the scalar and block local to global mappings */
    ierr = IGA_Grid_GetLGMap(grid,&iga->lgmap);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->lgmap);CHKERRQ(ierr);
    ierr = IGA_Grid_GetLGMapBlock(grid,&iga->lgmapb);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->lgmapb);CHKERRQ(ierr);
    /* build global <-> local vector scatters */
    ierr = IGA_Grid_GetScatterG2L(grid,&iga->g2l);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->g2l);CHKERRQ(ierr);
    ierr = IGA_Grid_GetScatterL2G(grid,&iga->l2g);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->l2g);CHKERRQ(ierr);
    /* build global <-> natural vector scatter */
    ierr = IGA_Grid_NewScatterApp(grid,
                                  iga->geom_sizes,iga->geom_lstart,iga->geom_lwidth,
                                  &iga->natural,&iga->n2g,&iga->g2n);CHKERRQ(ierr);
    /* destroy the grid context */
    ierr = IGA_Grid_Destroy(&grid);CHKERRQ(ierr);
  }

  if (!iga->vectype) {
    const MatType vtype = VECSTANDARD;
    ierr = IGASetVecType(iga,vtype);CHKERRQ(ierr);
  }
  if (!iga->mattype) {
    const MatType mtype = (iga->dof > 1) ? MATBAIJ : MATAIJ;
    ierr = IGASetMatType(iga,mtype);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

extern PetscErrorCode IGASetUp_Basic(IGA);
extern PetscErrorCode IGASetUp_View(IGA);

#undef  __FUNCT__
#define __FUNCT__ "IGASetUp_Basic"
PetscErrorCode IGASetUp_Basic(IGA iga)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  ierr = IGASetUp_Stage1(iga);;CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetUp_View"
PetscErrorCode IGASetUp_View(IGA iga)
{
  PetscBool      flg1,flg2,info=PETSC_FALSE;
  char           filename1[PETSC_MAX_PATH_LEN] = "";
  char           filename2[PETSC_MAX_PATH_LEN] = "";
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);

  ierr = PetscObjectOptionsBegin((PetscObject)iga);CHKERRQ(ierr);
  ierr = PetscOptionsString("-iga_view",        "Information on IGA context",      "IGAView",filename1,filename1,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsBool(  "-iga_view_info",   "Output more detailed information","IGAView",info,&info,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool(  "-iga_view_detail", "Output more detailed information","IGAView",info,&info,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-iga_view_binary", "Save to file in binary format",   "IGAView",filename2,filename2,PETSC_MAX_PATH_LEN,&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if ((flg1||info) && !PetscPreLoadingOn) {
    ierr = PetscViewerASCIIOpen(((PetscObject)iga)->comm,filename1,&viewer);CHKERRQ(ierr);
    if (info) {ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);}
    ierr = IGAView(iga,viewer);CHKERRQ(ierr);
    if (info) {ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);}
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  if (flg2 && !PetscPreLoadingOn) {
    if (filename2[0]) {
      ierr = IGAWrite(iga,filename2);CHKERRQ(ierr);
    } else {
      viewer = PETSC_VIEWER_BINARY_(((PetscObject)iga)->comm);
      ierr = IGAView(iga,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "IGASetUp"
/*@
   IGASetUp - Sets up the internal data structures for the later use
   of the IGA.

   Collective on IGA

   Input Parameter:
.  iga - the IGA context

   Level: normal

.keywords: IGA, options
@*/
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

  iga->setup = PETSC_TRUE;

  /* --- Stage 1 --- */
  ierr = IGASetUp_Stage1(iga);CHKERRQ(ierr);

  /* --- Stage 2 --- */
  ierr = IGASetUp_Stage2(iga);CHKERRQ(ierr);

  /* --- Stage 3 --- */
  iga->setupstage = 3;

  for (i=iga->dim; i<3; i++) {
    ierr = IGABoundaryReset(iga->boundary[i][0]);CHKERRQ(ierr);
    ierr = IGABoundaryReset(iga->boundary[i][1]);CHKERRQ(ierr);
  }

  if (iga->order < 0)
    for (i=0; i<iga->dim; i++)
      iga->order = PetscMax(iga->order,iga->axis[i]->p);
  iga->order = PetscMax(iga->order,1); /* XXX */
  iga->order = PetscMin(iga->order,3); /* XXX */

  for (i=0; i<3; i++) {
    if (i >= iga->dim) {ierr = IGARuleReset(iga->rule[i]);CHKERRQ(ierr);}
    if (iga->rule[i]->nqp < 1) {ierr = IGARuleInit(iga->rule[i],iga->axis[i]->p + 1);CHKERRQ(ierr);}
    ierr = IGABasisInitQuadrature(iga->basis[i],iga->axis[i],iga->rule[i],iga->order);CHKERRQ(ierr);
  }
  ierr = IGAElementInit(iga->iterator,iga);CHKERRQ(ierr);

  ierr = IGASetUp_View(iga);CHKERRQ(ierr);

  if (iga->collocation) {ierr = IGASetUseCollocation(iga,PETSC_TRUE);CHKERRQ(ierr);}  /* collocation */

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetVecType"
PetscErrorCode IGASetVecType(IGA iga,const VecType vectype)
{
  char           *vtype;
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
  char           *mtype;
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
#define __FUNCT__ "IGASetUserSystem"
/*@
   IGASetUserSystem - Set the user callback to form the matrix and vector
   which represents the discretized a(w,u) = L(w).

   Logically collective on IGA

   Input Parameters:
+  iga - the IGA context
.  System - the function which evaluates a(w,u) and L(w)
-  ctx - user-defined context for evaluation routine (may be PETSC_NULL)

   Details of System:
$  PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx);

+  p - point at which to evaluate a(w,u)=L(w)
.  K - contribution to a(w,u)
.  F - contribution to L(w)
-  ctx - user-defined context for evaluation routine

   Level: normal

.keywords: IGA, setup linear system, matrix assembly, vector assembly
@*/
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
#define __FUNCT__ "IGASetUserIFunction2"
/*@
   IGASetUserIFunction - Set the function which computes the residual
       F(t,U_tt,U_t,U)=0 for use with implicit time stepping routines.

   Logically Collective on TS

   Input Parameter:
+  iga - the IGA context
.  IFunction - the function evaluation routine
-  IFunCtx - user-defined context for private data for the function evaluation routine (may be PETSC_NULL)

   Details of IFunction:
$  PetscErrorCode IFunction(IGAPoint p,PetscReal dt,
                            PetscReal a,const PetscScalar *A,
                            PetscReal v,const PetscScalar *V,
                            PetscReal t,const PetscScalar *U,
                            PetscScalar *F,void *ctx);

+  p - point at which to compute the residual
.  dt - time step size
.  a - positive parameter which depends on the time integration method
.  A - second time derivative of the state vector
.  v - positive parameter which depends on the time integration method
.  V - time derivative of the state vector
.  t - time at step/stage being solved
.  U - state vector
.  F - function vector
-  ctx - [optional] user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetUserIFunction2(IGA iga,IGAUserIFunction2 IFunction,void *IFunCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (IFunction) iga->userops->IFunction2 = IFunction;
  if (IFunCtx)   iga->userops->IFunCtx    = IFunCtx;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetUserIJacobian2"
/*@
   IGASetUserIJacobian2 - Set the function to compute the matrix
       J = a*dF/dU_tt + v*dF/dU_t + dF/dU  where F(t,U_tt,U_t,U) is
       the function you provided with IGASetUserIFunction2().

   Logically Collective on TS

   Input Parameter:
+  iga       - the IGA context
.  IJacobian - the Jacobian evaluation routine
-  IJacCtx   - user-defined context for private data for the Jacobian evaluation routine (may be PETSC_NULL)

   Details of IJacobian:
$  PetscErrorCode IJacobian(IGAPoint p,PetscReal dt,
                            PetscReal a,const PetscScalar *A,
                            PetscReal v,const PetscScalar *V,
                            PetscReal t,const PetscScalar *U,
                            PetscScalar *J,void *ctx);

+  p   - point at which to compute the Jacobian
.  dt  - time step size
.  a   - positive parameter which depends on the time integration method
.  A   - second time derivative of the state vector
.  v   - positive parameter which depends on the time integration method
.  V   - time derivative of the state vector
.  t   - time at step/stage being solved
.  U   - state vector
.  J   - Jacobian matrix
-  ctx - [optional] user-defined context for evaluation routine

   Level: normal

.keywords: IGA, options
@*/
PetscErrorCode IGASetUserIJacobian2(IGA iga,IGAUserIJacobian2 IJacobian,void *IJacCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (IJacobian) iga->userops->IJacobian2 = IJacobian;
  if (IJacCtx)   iga->userops->IJacCtx    = IJacCtx;
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


#undef  __FUNCT__
#define __FUNCT__ "IGAGetMeshInformation"
/*@
   IGAGetMeshInformation - Returns information concerning the mesh being used.

   Logically Collective on IGA

   Input Parameter:
+  iga - the IGA context

   Output Parameters:
+  hmin - minimum measure of h among all elements
.  hmax - maximum measure of h among all elements
.  havg - average measure of h among all elements
-  hsdt - standard deviation measure of h among all elements

   Level: normal

   Notes:
   This routine takes statistics on the output of
   IGAElementCharacteristicSize for all elements in all partitions of
   the mesh.

.keywords: IGA, mesh information
@*/
PetscErrorCode IGAGetMeshInformation(IGA iga,PetscReal *hmin,PetscReal *hmax,PetscReal *havg,PetscReal *hstd)
{
  PetscFunctionBegin;
  PetscReal      h,local_hmin,local_hmax,local_hsum,local_hsum2;
  PetscInt       i,nel;
  IGAElement     element;
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  IGACheckSetUp(iga,1);

  local_hmin  = PETSC_MAX_REAL;
  local_hmax  = 0;
  local_hsum  = 0;
  local_hsum2 = 0;
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementCharacteristicSize(element,&h);CHKERRQ(ierr);
    local_hmin  = PetscMin(local_hmin,h);
    local_hmax  = PetscMax(local_hmax,h);
    local_hsum  += h;
    local_hsum2 += h*h;
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&local_hmin, hmin,1,MPIU_REAL,MPIU_MIN,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&local_hmax, hmax,1,MPIU_REAL,MPIU_MAX,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&local_hsum, havg,1,MPIU_REAL,MPIU_SUM,comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&local_hsum2,hstd,1,MPIU_REAL,MPIU_SUM,comm);CHKERRQ(ierr);

  nel = 1;
  for(i=0;i<iga->dim;i++) nel *= iga->elem_sizes[i];
  *havg /= nel;
  *hstd = *hstd/nel - (*havg)*(*havg);

  PetscFunctionReturn(0);
}
