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
  *_iga = NULL;
  ierr = IGAInitializePackage();CHKERRQ(ierr);
#if PETSC_VERSION_LE(3,3,0)
  ierr = PetscHeaderCreate(iga,_p_IGA,struct _IGAOps,IGA_CLASSID,-1,
                           "IGA","IGA","IGA",comm,IGADestroy,IGAView);CHKERRQ(ierr);
#else
  ierr = PetscHeaderCreate(iga,_p_IGA,struct _IGAOps,IGA_CLASSID,
                           "IGA","IGA","IGA",comm,IGADestroy,IGAView);CHKERRQ(ierr);
#endif

  *_iga = iga;

  iga->vectype = NULL;
  iga->mattype = NULL;

  iga->dim = -1;
  iga->dof = -1;
  iga->order = -1;

  for (i=0; i<3; i++)
    iga->proc_sizes[i] = -1;

  for (i=0; i<3; i++) {
    ierr = IGAAxisCreate(&iga->axis[i]);CHKERRQ(ierr);
    ierr = IGARuleCreate(&iga->rule[i]);CHKERRQ(ierr);
    ierr = IGABasisCreate(&iga->basis[i]);CHKERRQ(ierr);
  }
  ierr = IGAFormCreate(&iga->form);CHKERRQ(ierr);
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
  }
  ierr = IGAFormDestroy(&iga->form);CHKERRQ(ierr);
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
  iga->setupstage = 0;

  /* element */
  ierr = DMDestroy(&iga->elem_dm);CHKERRQ(ierr);
  /* geometry */
  iga->rational = PETSC_FALSE;
  iga->geometry = 0;
  iga->property = 0;
  iga->fixtable = PETSC_FALSE;
  ierr = PetscFree(iga->rationalW);CHKERRQ(ierr);
  ierr = PetscFree(iga->geometryX);CHKERRQ(ierr);
  ierr = PetscFree(iga->propertyA);CHKERRQ(ierr);
  ierr = PetscFree(iga->fixtableU);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->geom_dm);CHKERRQ(ierr);
  /* node */
  ierr = AODestroy(&iga->ao);CHKERRQ(ierr);
  ierr = AODestroy(&iga->aob);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&iga->lgmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&iga->lgmapb);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&iga->map);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->g2l);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->l2g);CHKERRQ(ierr);
  ierr = VecDestroy(&iga->natural);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->n2g);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->g2n);CHKERRQ(ierr);
  while (iga->nwork > 0)
    {ierr = VecDestroy(&iga->vwork[--iga->nwork]);CHKERRQ(ierr);}
  ierr = DMDestroy(&iga->node_dm);CHKERRQ(ierr);

  ierr = IGAElementReset(iga->iterator);CHKERRQ(ierr);

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
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL)
      {
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

#if PETSC_VERSION_LE(3,3,0)
PETSC_EXTERN PetscErrorCode PetscOptionsGetViewer(MPI_Comm,const char[],const char[],PetscViewer*,PetscViewerFormat*,PetscBool*);
#endif

#undef  __FUNCT__
#define __FUNCT__ "IGAViewFromOptions"
PetscErrorCode IGAViewFromOptions(IGA iga,const char prefix[],const char option[])
{

  MPI_Comm          comm;
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         skipinfo = PETSC_FALSE;
  PetscBool         flg;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  if (!prefix) {ierr = IGAGetOptionsPrefix(iga,&prefix);CHKERRQ(ierr);}
  ierr = PetscOptionsGetBool(NULL,"-viewer_binary_skip_info",&skipinfo,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-viewer_binary_skip_info","");CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(comm,prefix,option,&viewer,&format,&flg);CHKERRQ(ierr);
  if (!skipinfo) {ierr = PetscOptionsClearValue("-viewer_binary_skip_info");CHKERRQ(ierr);}
  if (!flg) PetscFunctionReturn(0);
  ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
  ierr = IGAView(iga,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
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
    ierr = PetscMalloc1(iga->dof+1,&iga->fieldname);CHKERRQ(ierr);
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
/*@
   IGASetOrder - Sets the maximum available derivative order
   for basis and shape functions.

   Logically Collective on IGA

   Input Parameters:
+  iga - the IGA context
-  order - the maximum available derivative order

   Notes:
   Currently, the maximum derivative order to compute is at least one
   and at most three. The default value is determined as the maximum
   polynomial degree over the parametric directions.

   Level: normal

.keywords: IGA, order
@*/
PetscErrorCode IGASetOrder(IGA iga,PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,order,2);
  if (order < 0)
    SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
             "Order must be nonnegative, got %D",order);
  if (iga->order == order) PetscFunctionReturn(0);
  iga->order = order;
  iga->setup = PETSC_FALSE;
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
#define __FUNCT__ "IGASetBasisType"
PetscErrorCode IGASetBasisType(IGA iga,PetscInt i,IGABasisType type)
{
  IGABasis       basis;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,i,2);
  PetscValidLogicalCollectiveInt(iga,(PetscInt)type,3);
  ierr = IGAGetBasis(iga,i,&basis);CHKERRQ(ierr);
  if (basis->type == type) PetscFunctionReturn(0);
  ierr = IGABasisSetType(basis,type);CHKERRQ(ierr);
  iga->setup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetQuadrature"
PetscErrorCode IGASetQuadrature(IGA iga,PetscInt i,PetscInt q)
{
  IGARule        rule;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,i,2);
  PetscValidLogicalCollectiveInt(iga,q,3);
  ierr = IGAGetRule(iga,i,&rule);CHKERRQ(ierr);
  if (q == PETSC_DECIDE && iga->axis[i]->p > 0) q = iga->axis[i]->p + 1;
  if (q <= 0) SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of quadrature points %D must be positive",q);
  if (rule->nqp == q) PetscFunctionReturn(0);
  ierr = IGARuleInit(rule,q);CHKERRQ(ierr);
  iga->setup = PETSC_FALSE;
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
  PetscValidLogicalCollectiveInt(iga,i,2);
  PetscValidLogicalCollectiveInt(iga,processors,3);
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveBool(iga,collocation,2);
  if (iga->setupstage > 0 && iga->collocation != collocation)
    SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
            "Cannot change collocation after IGASetUp()");
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
#define __FUNCT__ "IGAGetBasis"
PetscErrorCode IGAGetBasis(IGA iga,PetscInt i,IGABasis *basis)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(basis,3);
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

PETSC_STATIC_INLINE
#undef  __FUNCT__
#define __FUNCT__ "IGAOptionsReject"
PetscErrorCode IGAOptionsReject(const char prefix[],const char name[])
{
  PetscErrorCode ierr;
  PetscBool      flag = PETSC_FALSE;
  PetscFunctionBegin;
  ierr = PetscOptionsHasName(prefix,name,&flag);CHKERRQ(ierr);
  if (flag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Disabled option: %s",name);
  PetscFunctionReturn(0);
}

#if PETSC_VERSION_LT(3,5,0)
#define PetscOptionsFList PetscOptionsList
#endif

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
    const char *prefix = 0;
    PetscBool collocation = iga->collocation;
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

    ierr = IGAGetOptionsPrefix(iga,&prefix);CHKERRQ(ierr);

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
    if (iga->setupstage) goto setupcalled;

    ierr = PetscOptionsInt("-iga_dim","Number of dimensions","IGASetDim",iga->dim,&dim,&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetDim(iga,dim);CHKERRQ(ierr);}
    dim = (iga->dim > 0) ? iga->dim : 3;

    ierr = PetscOptionsInt("-iga_dof","Number of DOFs per node","IGASetDof",iga->dof,&dof,&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetDof(iga,dof);CHKERRQ(ierr);}
    dof = (iga->dof > 0) ? iga->dof : 1;

    /* Collocation */
    ierr = PetscOptionsBool("-iga_collocation","Use collocation","IGASetUseCollocation",collocation,&collocation,&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetUseCollocation(iga,collocation);CHKERRQ(ierr);}
    if (iga->collocation) {ierr = IGAOptionsReject(prefix,"-iga_quadrature");CHKERRQ(ierr);}

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

    /* Geometry */
    ierr = PetscOptionsString("-iga_load","Specify IGA geometry file","IGARead",filename,filename,sizeof(filename),&flg);CHKERRQ(ierr);
    if (!flg) {ierr = PetscOptionsString("-iga_geometry","deprecated, use -iga_load","IGARead",filename,filename,sizeof(filename),&flg);CHKERRQ(ierr);}
    if (flg) { /* load from file */
      ierr = IGAOptionsReject(prefix,"-iga_elements"  );CHKERRQ(ierr);
      ierr = IGAOptionsReject(prefix,"-iga_degree"    );CHKERRQ(ierr);
      ierr = IGAOptionsReject(prefix,"-iga_continuity");CHKERRQ(ierr);
      ierr = IGAOptionsReject(prefix,"-iga_limits"    );CHKERRQ(ierr);
      ierr = IGARead(iga,filename);CHKERRQ(ierr);
      ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    } else { /* set axis details */
      ierr = PetscOptionsEnum("-iga_basis_type","Basis type","IGASetBasisType",IGABasisTypes,(PetscEnum)btype[0],(PetscEnum*)&btype[0],&flg);CHKERRQ(ierr);
      if (flg) for (i=1; i<dim; i++) btype[i] = btype[0]; /* XXX */
      for (i=0; i<dim; i++) if (btype[i] != IGA_BASIS_BSPLINE) conts[i] = 0;
      ierr = PetscOptionsIntArray ("-iga_elements",  "Elements",  "IGAAxisInitUniform",elems,(ne=dim,&ne),&flg);CHKERRQ(ierr);
      ierr = PetscOptionsIntArray ("-iga_degree",    "Degree",    "IGAAxisSetDegree",  degrs,(nd=dim,&nd),NULL);CHKERRQ(ierr);
      ierr = PetscOptionsIntArray ("-iga_continuity","Continuity","IGAAxisInitUniform",conts,(nc=dim,&nc),NULL);CHKERRQ(ierr);
      ierr = PetscOptionsRealArray("-iga_limits",    "Limits",    "IGAAxisInitUniform",&ulims[0][0],(nl=2*dim,&nl),NULL);CHKERRQ(ierr);
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

  setupcalled:

    /* Order */
    if (order < 0) for (i=0; i<dim; i++) order = PetscMax(order,iga->axis[i]->p);
    order = PetscMax(order,1); order = PetscMin(order,3);
    ierr = PetscOptionsInt("-iga_order","Maximum available derivative order","IGASetOrder",order,&order,&flg);CHKERRQ(ierr);
    if (flg) { ierr = IGASetOrder(iga,order);CHKERRQ(ierr);}

    /* Quadrature */
    for (i=0; i<dim; i++) if (quadr[i] < 1) quadr[i] = iga->axis[i]->p + 1;
    ierr = PetscOptionsIntArray("-iga_quadrature","Quadrature points","IGASetQuadrature",quadr,(nq=dim,&nq),&flg);CHKERRQ(ierr);
    if (flg) for (i=0; i<dim; i++) {
        PetscInt q = (i<nq) ? quadr[i] : quadr[0];
        if (q > 0) {ierr = IGASetQuadrature(iga,i,q);CHKERRQ(ierr);}
      }

    /* Basis Type */
    ierr = PetscOptionsEnum("-iga_basis_type","Basis type","IGASetBasisType",IGABasisTypes,(PetscEnum)btype[0],(PetscEnum*)&btype[0],&flg);CHKERRQ(ierr);
    if (flg) for (i=1; i<dim; i++) btype[i] = btype[0]; /* XXX */
    if (flg) for (i=0; i<dim; i++) {
        ierr = IGASetBasisType(iga,i,btype[i]);CHKERRQ(ierr);
      }

    /* Matrix and Vector type */
    if (iga->dof == 1) {ierr = PetscStrcpy(mtype,MATAIJ);CHKERRQ(ierr);}
    if (iga->vectype)  {ierr = PetscStrncpy(vtype,iga->vectype,sizeof(vtype));CHKERRQ(ierr);}
    if (iga->mattype)  {ierr = PetscStrncpy(mtype,iga->mattype,sizeof(mtype));CHKERRQ(ierr);}
    ierr = PetscOptionsFList("-iga_vec_type","Vector type","IGASetVecType",VecList,vtype,vtype,sizeof(vtype),&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetVecType(iga,vtype);CHKERRQ(ierr);}
    ierr = PetscOptionsFList("-iga_mat_type","Matrix type","IGASetMatType",MatList,mtype,mtype,sizeof(mtype),&flg);CHKERRQ(ierr);
    if (flg) {ierr = IGASetMatType(iga,mtype);CHKERRQ(ierr);}

    /* View options, handled in IGASetUp() */
    ierr = PetscOptionsName("-iga_view",       "Information on IGA context",      "IGAView",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-iga_view_ascii", "Information on IGA context",      "IGAView",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-iga_view_info",  "Output more detailed information","IGAView",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-iga_view_detail","Output more detailed information","IGAView",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsName("-iga_view_binary","Save to file in binary format",   "IGAView",NULL);CHKERRQ(ierr);

    ierr = PetscObjectProcessOptionsHandlers((PetscObject)iga);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAOptionsAlias"
PetscErrorCode IGAOptionsAlias(const char name[],const char defval[],const char alias[])
{
  char           value[4096]= {0};
  PetscBool      flag = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidCharPointer(name,1);
  PetscValidCharPointer(alias,3);
  ierr = PetscOptionsHasName(NULL,name,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PetscOptionsGetString(NULL,name,value,sizeof(value),&flag);CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(alias,value);CHKERRQ(ierr);
  } else if (defval) {
    ierr = PetscOptionsHasName(NULL,alias,&flag);CHKERRQ(ierr);
    if (!flag) {ierr = PetscOptionsSetValue(alias,defval);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode IGACreateSubComms1D(IGA,MPI_Comm[]);

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
  PetscInt         *ranges[3] = {NULL, NULL, NULL};
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
    ierr = PetscMalloc1(procs[i],&ranges[i]);CHKERRQ(ierr);
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
                     NULL,PETSC_TRUE,0,dm);CHKERRQ(ierr);
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
                     NULL,PETSC_TRUE,0,dm);CHKERRQ(ierr);
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
                     NULL,PETSC_TRUE,0,dm);CHKERRQ(ierr);
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

EXTERN_C_BEGIN
extern PetscReal IGA_Greville(PetscInt i,PetscInt p,const PetscReal U[]);
extern PetscInt  IGA_FindSpan(PetscInt n,PetscInt p,PetscReal u, const PetscReal U[]);
EXTERN_C_END

#undef  __FUNCT__
#define __FUNCT__ "IGASetUp_Stage1"
static PetscErrorCode IGASetUp_Stage1(IGA iga)
{
  PetscInt       i,dim;
  PetscInt       grid_sizes[3] = {1,1,1};
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (iga->setupstage >= 1) PetscFunctionReturn(0);
  iga->setupstage = 1;

  dim = iga->dim;

  for (i=0; i<dim; i++)
    {ierr = IGAAxisSetUp(iga->axis[i]);CHKERRQ(ierr);}
  for (i=dim; i<3; i++)
    {ierr = IGAAxisReset(iga->axis[i]);CHKERRQ(ierr);}

  if (!iga->collocation)
    for (i=0; i<dim; i++)
      grid_sizes[i] = iga->axis[i]->nel;
  else
    for (i=0; i<dim; i++)
      grid_sizes[i] = iga->axis[i]->nnp;

  { /* processor grid and coordinates */
    MPI_Comm    comm = ((PetscObject)iga)->comm;
    PetscMPIInt size,rank;
    PetscInt    *proc_sizes = iga->proc_sizes;
    PetscInt    *proc_ranks = iga->proc_ranks;
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = IGA_Partition(size,rank,iga->dim,grid_sizes,
                         proc_sizes,proc_ranks);CHKERRQ(ierr);
    for (i=dim; i<3; i++) {
      proc_sizes[i] = 1;
      proc_ranks[i] = 0;
    }
  }

  { /* element partitioning */
    PetscInt *elem_sizes = iga->elem_sizes;
    PetscInt *elem_start = iga->elem_start;
    PetscInt *elem_width = iga->elem_width;
    for (i=0; i<dim; i++) elem_sizes[i] = grid_sizes[i];
    ierr = IGA_Distribute(iga->dim,iga->proc_sizes,iga->proc_ranks,
                          elem_sizes,elem_width,elem_start);CHKERRQ(ierr);
    for (i=dim; i<3; i++) {
      elem_sizes[i] = 1;
      elem_start[i] = 0;
      elem_width[i] = 1;
    }
  }

  if (!iga->collocation)
  { /* geometry partitioning */
    PetscInt *geom_sizes  = iga->geom_sizes;
    PetscInt *geom_lstart = iga->geom_lstart;
    PetscInt *geom_lwidth = iga->geom_lwidth;
    PetscInt *geom_gstart = iga->geom_gstart;
    PetscInt *geom_gwidth = iga->geom_gwidth;
    for (i=0; i<dim; i++) {
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
    for (i=dim; i<3; i++) {
      geom_sizes[i]  = 1;
      geom_lstart[i] = 0;
      geom_lwidth[i] = 1;
      geom_gstart[i] = 0;
      geom_gwidth[i] = 1;
    }
  } else
  {  /* geometry partitioning */
    PetscInt *geom_sizes  = iga->geom_sizes;
    PetscInt *geom_lstart = iga->geom_lstart;
    PetscInt *geom_lwidth = iga->geom_lwidth;
    PetscInt *geom_gstart = iga->geom_gstart;
    PetscInt *geom_gwidth = iga->geom_gwidth;
    for (i=0; i<dim; i++) {
      PetscInt   p = iga->axis[i]->p;
      PetscInt   m = iga->axis[i]->m;
      PetscReal *U = iga->axis[i]->U;
      PetscInt   n = m - p - 1;
      geom_sizes[i]  = iga->elem_sizes[i];
      geom_lstart[i] = iga->elem_start[i];
      geom_lwidth[i] = iga->elem_width[i];
      {
        PetscInt  a = geom_lstart[i];
        PetscReal u = IGA_Greville(a,p,U);
        PetscInt  k = IGA_FindSpan(n,p,u,U);
        geom_gstart[i] = k - p;
      }
      {
        PetscInt  a = geom_lstart[i] + geom_lwidth[i] - 1;
        PetscReal u = IGA_Greville(a,p,U);
        PetscInt  k = IGA_FindSpan(n,p,u,U);
        geom_gwidth[i] = k + 1 - geom_gstart[i];
      }
    }
    for (i=dim; i<3; i++) {
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
  iga->rational = PETSC_FALSE;
  iga->geometry = 0;
  iga->property = 0;
  iga->fixtable = PETSC_FALSE;
  ierr = PetscFree(iga->rationalW);CHKERRQ(ierr);
  ierr = PetscFree(iga->geometryX);CHKERRQ(ierr);
  ierr = PetscFree(iga->propertyA);CHKERRQ(ierr);
  ierr = PetscFree(iga->fixtableU);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->geom_dm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetUp_Stage2"
static PetscErrorCode IGASetUp_Stage2(IGA iga)
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
      if (rank == 0 && node_lstart[i] < 0) {
        node_lwidth[i] += node_lstart[i];
        node_lstart[i] = 0;
      }
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
  ierr = PetscLayoutDestroy(&iga->map);CHKERRQ(ierr);
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
    /* build the layout */
    ierr = IGA_Grid_GetLayout(grid,&iga->map);CHKERRQ(ierr);
    ierr = PetscLayoutReference(iga->map,&iga->map);CHKERRQ(ierr);
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

PETSC_EXTERN PetscErrorCode IGASetUp_Basic(IGA);

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
static PetscErrorCode IGASetUp_View(IGA iga)
{
  PetscBool      flg1,flg2,info=PETSC_FALSE;
  char           filename1[PETSC_MAX_PATH_LEN] = "";
  char           filename2[PETSC_MAX_PATH_LEN] = "";
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);

  ierr = PetscObjectOptionsBegin((PetscObject)iga);CHKERRQ(ierr);
  ierr = PetscOptionsString("-iga_view_ascii",  "Information on IGA context",      "IGAView",filename1,filename1,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsBool(  "-iga_view_info",   "Output more detailed information","IGAView",info,&info,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool(  "-iga_view_detail", "Output more detailed information","IGAView",info,&info,NULL);CHKERRQ(ierr);
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

  if (iga->order < 0)
    for (i=0; i<iga->dim; i++)
      iga->order = PetscMax(iga->order,iga->axis[i]->p);
  iga->order = PetscMax(iga->order,1); /* XXX */
  iga->order = PetscMin(iga->order,3); /* XXX */

  if (iga->collocation) {
    for (i=0; i<iga->dim; i++)
      if(iga->axis[i]->periodic)
        SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_SUP,"Collocation not supported with periodicity");
  }

  for (i=0; i<3; i++)
    if (!iga->collocation) {
      if (i >= iga->dim || iga->rule[i]->nqp < 1)
        {ierr = IGARuleInit(iga->rule[i],iga->axis[i]->p + 1);CHKERRQ(ierr);}
    }
    else
      {ierr = IGARuleReset(iga->rule[i]);CHKERRQ(ierr);}

  for (i=0; i<3; i++)
    if (!iga->collocation)
      {ierr = IGABasisInitQuadrature(iga->basis[i],iga->axis[i],iga->rule[i],iga->order);CHKERRQ(ierr);}
    else
      {ierr = IGABasisInitCollocation(iga->basis[i],iga->axis[i],iga->order);CHKERRQ(ierr);}

  ierr = IGAElementInit(iga->iterator,iga);CHKERRQ(ierr);

  ierr = IGAViewFromOptions(iga,NULL,"-iga_view");CHKERRQ(ierr);
  ierr = IGASetUp_View(iga);CHKERRQ(ierr);
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
