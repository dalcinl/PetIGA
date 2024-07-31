#include "petiga.h"
#include "petigagrid.h"

static const char citation[] =
"@article{PetIGA,\n"
" author = {L. Dalcin and N. Collier and P. Vignal and A.M.A. C\\^{o}rtes and V.M. Calo},\n"
" title = {{PetIGA}: A framework for high-performance isogeometric analysis},\n"
" journal = {Computer Methods in Applied Mechanics and Engineering},\n"
" volume = {308},\n"
" pages = {151--181},\n"
" year = {2016},\n"
" issn = {0045-7825},\n"
" doi = {https://doi.org/10.1016/j.cma.2016.05.011},\n"
"}\n";
static PetscBool cited = PETSC_FALSE;

/*@
   IGACreate - Creates the default IGA context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  newiga - location to put the IGA context

   Level: normal

.keywords: IGA, create
@*/
PetscErrorCode IGACreate(MPI_Comm comm,IGA *newiga)
{
  PetscInt       i;
  IGA            iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscAssertPointer(newiga,2);

  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);
  ierr = IGAInitializePackage();CHKERRQ(ierr);

  *newiga = NULL;
  ierr = PetscHeaderCreate(iga,IGA_CLASSID,"IGA","IGA","IGA",comm,IGADestroy,IGAView);CHKERRQ(ierr);
  *newiga = iga;

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
  PetscAssertPointer(_iga,1);
  iga = *_iga; *_iga = NULL;
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

PetscErrorCode IGAReset(IGA iga)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!iga) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);

  iga->setup = PETSC_FALSE;
  iga->setupstage = 0;

  /* geometry */
  iga->rational = PETSC_FALSE;
  iga->geometry = 0;
  iga->property = 0;
  ierr = PetscFree(iga->rationalW);CHKERRQ(ierr);
  ierr = PetscFree(iga->geometryX);CHKERRQ(ierr);
  ierr = PetscFree(iga->propertyA);CHKERRQ(ierr);
  /* fixtable */
  iga->fixtable = PETSC_FALSE;
  ierr = PetscFree(iga->fixtableU);CHKERRQ(ierr);
  /* node */
  ierr = AODestroy(&iga->ao);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&iga->lgmap);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&iga->map);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->g2l);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->l2g);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->l2l);CHKERRQ(ierr);
  ierr = VecDestroy(&iga->natural);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->n2g);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->g2n);CHKERRQ(ierr);
  while (iga->nwork > 0)
    {ierr = VecDestroy(&iga->vwork[--iga->nwork]);CHKERRQ(ierr);}

  ierr = DMDestroy(&iga->geom_dm);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->elem_dm);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->node_dm);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->draw_dm);CHKERRQ(ierr);

  ierr = IGAElementReset(iga->iterator);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscInt IGA_NextKnot(PetscInt m,const PetscReal U[],PetscInt k,PetscInt direction);
EXTERN_C_END

static inline
void IGA_ContinuityString(IGAAxis axis,char buf[8],size_t len)
{
  PetscInt  p  = axis->p;
  PetscInt  m  = axis->m;
  PetscReal *U = axis->U;
  PetscInt Cmin=p,Cmax=-1;
  PetscInt k,j,ks=p+1,ke=m-p;
  if (axis->periodic) ks = IGA_NextKnot(m,U,p,-1)+1;
  if (axis->periodic) ke = IGA_NextKnot(m,U,ke,+1)-1;
  for (k=ks; k<ke; k=j) {
    j = IGA_NextKnot(m,U,k,1);
    Cmin = PetscMin(Cmin,p-(j-k));
    Cmax = PetscMax(Cmax,p-(j-k));
  }
  (void)PetscSNPrintf(buf,len,(Cmin==Cmax)?"%d":"%d:%d",(int)Cmin,(int)Cmax);
  if (axis->nel==1 && !axis->periodic) (void)PetscStrcpy(buf,"*");
}

PetscErrorCode IGAPrint(IGA iga,PetscViewer viewer)
{
  PetscBool      match;
  PetscInt       i,dim,dof,order;
  const char     *name = NULL;
  const char     *prefix = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(iga,1,viewer,2);
  IGACheckSetUp(iga,1);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&match);CHKERRQ(ierr);
  if (!match) PetscFunctionReturn(0);

  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);
  ierr = IGAGetOrder(iga,&order);CHKERRQ(ierr);
  ierr = IGAGetName(iga,&name);CHKERRQ(ierr);
  ierr = IGAGetOptionsPrefix(iga,&prefix);CHKERRQ(ierr);
  if (name || prefix) {ierr = PetscViewerASCIIPrintf(viewer,"IGA: name=%s prefix=%s\n",name,prefix);CHKERRQ(ierr);}
  ierr = PetscViewerASCIIPrintf(viewer,"IGA: dim=%d dof=%d order=%d geometry=%d rational=%d property=%d\n",
                                (int)dim,(int)dof,(int)order,(int)iga->geometry,(int)iga->rational,(int)iga->property);CHKERRQ(ierr);
  for (i=0; i<dim; i++) {
    char Cbuf[8]; IGA_ContinuityString(iga->axis[i],Cbuf,sizeof(Cbuf));
    ierr = PetscViewerASCIIPrintf(viewer,"Axis %d: basis=%s[%d,%s] rule=%s[%d] periodic=%d nnp=%d nel=%d\n",(int)i,
                                  IGABasisTypes[iga->basis[i]->type],(int)iga->axis[i]->p,Cbuf,
                                  IGARuleTypes[iga->rule[i]->type],(int)iga->rule[i]->nqp,
                                  (int)iga->axis[i]->periodic,(int)iga->node_sizes[i],(int)iga->elem_sizes[i]);CHKERRQ(ierr);
  }
  {
    MPI_Comm comm; PetscViewerFormat format;
    PetscInt *sizes = iga->proc_sizes, size = sizes[0]*sizes[1]*sizes[2];
    PetscInt isum[2],imin[2],imax[2],ival[2] = {1,1};
    for (i=0; i<dim; i++) {ival[0] *= iga->node_lwidth[i]; ival[1] *= iga->elem_width[i];}
    ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(ival,isum,2,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(ival,imin,2,MPIU_INT,MPI_MIN,comm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(ival,imax,2,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Partition - MPI: processors=[%d,%d,%d] total=%d\n",
                                  (int)sizes[0],(int)sizes[1],(int)sizes[2],(int)size);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Partition - nnp: sum=%" PetscInt_FMT " min=%" PetscInt_FMT " max=%" PetscInt_FMT " max/min=%g\n",
                                  isum[0],imin[0],imax[0],(double)imax[0]/(double)imin[0]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Partition - nel: sum=%" PetscInt_FMT " min=%" PetscInt_FMT " max=%" PetscInt_FMT " max/min=%g\n",
                                  isum[1],imin[1],imax[1],(double)imax[1]/(double)imin[1]);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscMPIInt rank; PetscInt *ranks = iga->proc_ranks;
      PetscInt *nnp = iga->node_lwidth, tnnp = 1, *snp = iga->node_lstart;
      PetscInt *nel = iga->elem_width,  tnel = 1, *sel = iga->elem_start;
      for (i=0; i<dim; i++) {tnnp *= nnp[i]; tnel *= nel[i];}
      ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] (%d,%d,%d): ",
                                                (int)rank,(int)ranks[0],(int)ranks[1],(int)ranks[2]);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"nnp=[%d:%d|%d:%d|%d:%d]=[%d|%d|%d]=%" PetscInt_FMT "  ",
                                                (int)snp[0],(int)snp[0]+(int)nnp[0]-1,
                                                (int)snp[1],(int)snp[1]+(int)nnp[1]-1,
                                                (int)snp[2],(int)snp[2]+(int)nnp[2]-1,
                                                (int)nnp[0],(int)nnp[1],(int)nnp[2],tnnp);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"nel=[%d:%d|%d:%d|%d:%d]=[%d|%d|%d]=%" PetscInt_FMT "\n",
                                                (int)sel[0],(int)sel[0]+(int)nel[0]-1,
                                                (int)sel[1],(int)sel[1]+(int)nel[1]-1,
                                                (int)sel[2],(int)sel[2]+(int)nel[2]-1,
                                                (int)nel[0],(int)nel[1],(int)nel[2],tnel);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#if PETSC_VERSION_LT(3,8,0)
#define PETSCVIEWERGLVIS "glvis"
#endif

PetscErrorCode IGAView(IGA iga,PetscViewer viewer)
{
  PetscBool         match;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(((PetscObject)iga)->comm,&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(iga,1,viewer,2);
  if (!iga->setup) PetscFunctionReturn(0); /* XXX */

#if PETSC_VERSION_LT(3,14,0)
{
  PetscViewerFormat format;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&match);CHKERRQ(ierr);
  if (match && format == PETSC_VIEWER_ASCII_VTK) { ierr = IGADraw(iga,viewer);CHKERRQ(ierr); PetscFunctionReturn(0); }
}
#endif
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&match);CHKERRQ(ierr);
  if (match) { ierr = IGASave(iga,viewer);CHKERRQ(ierr); PetscFunctionReturn(0); }
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&match);CHKERRQ(ierr);
  if (match) { ierr = IGAPrint(iga,viewer);CHKERRQ(ierr); PetscFunctionReturn(0); }
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&match);CHKERRQ(ierr);
  if (match) { ierr = IGADraw(iga,viewer);CHKERRQ(ierr); PetscFunctionReturn(0); }
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERVTK,&match);CHKERRQ(ierr);
  if (match) { ierr = IGADraw(iga,viewer);CHKERRQ(ierr); PetscFunctionReturn(0); }
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERGLVIS,&match);CHKERRQ(ierr);
  if (match) { ierr = IGADraw(iga,viewer);CHKERRQ(ierr); PetscFunctionReturn(0); }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAViewFromOptions(IGA iga,PetscObject obj,const char name[])
{

  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)iga,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetComm(IGA iga,MPI_Comm *comm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscAssertPointer(comm,2);
  *comm = ((PetscObject)iga)->comm;
  PetscFunctionReturn(0);
}

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
    SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of parametric dimensions must be in range [1,3], got %d",(int)dim);
  if (iga->dim > 0 && iga->dim != dim)
    SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot change IGA dim from %d after it was set to %d",(int)iga->dim,(int)dim);
  iga->dim = dim;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetDim(IGA iga,PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscAssertPointer(dim,2);
  *dim = iga->dim;
  PetscFunctionReturn(0);
}

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
    SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of DOFs per node must be greater than one, got %d",(int)dof);
  if (iga->dof > 0 && iga->dof != dof)
    SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot change number of DOFs from %d after it was set to %d",(int)iga->dof,(int)dof);
  iga->dof = dof;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetDof(IGA iga,PetscInt *dof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscAssertPointer(dof,2);
  *dof = iga->dof;
  PetscFunctionReturn(0);
}

PetscErrorCode IGASetName(IGA iga,const char name[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (name) PetscAssertPointer(name,2);
  ierr = PetscObjectSetName((PetscObject)iga,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetName(IGA iga,const char *name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscAssertPointer(name,2);
  *name = ((PetscObject)iga)->name;
  PetscFunctionReturn(0);
}

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
  PetscAssertPointer(name,3);
  if (iga->dof < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetDof() first");
  if (field < 0 || field >= iga->dof)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Field number must be in range [0,%d], got %d",(int)iga->dof-1,(int)field);
  if (!iga->fieldname) {ierr = PetscCalloc1((size_t)(iga->dof+1),&iga->fieldname);CHKERRQ(ierr);}
  ierr = PetscStrallocpy(name,&fname);CHKERRQ(ierr);
  ierr = PetscFree(iga->fieldname[field]);CHKERRQ(ierr);
  iga->fieldname[field] = fname;
  if (iga->node_dm) {ierr = DMDASetFieldName(iga->node_dm,field,fname);CHKERRQ(ierr);}
  if (iga->draw_dm) {ierr = DMDASetFieldName(iga->draw_dm,field,fname);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetFieldName(IGA iga,PetscInt field,const char *name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscAssertPointer(name,3);
  if (iga->dof < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetDof() first");
  if (field < 0 || field >= iga->dof)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Field number must be in range [0,%d], got %d",(int)iga->dof-1,(int)field);
  *name = iga->fieldname ? iga->fieldname[field] : NULL;
  PetscFunctionReturn(0);
}

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
    SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Order must be nonnegative, got %d",(int)order);
  iga->order = PetscClipInterval(order,1,4);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetOrder(IGA iga,PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscAssertPointer(order,2);
  *order = iga->order;
  PetscFunctionReturn(0);
}

PetscErrorCode IGASetBasisType(IGA iga,PetscInt i,IGABasisType type)
{
  IGABasis       basis;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,i,2);
  PetscValidLogicalCollectiveEnum(iga,type,3);
  if (iga->collocation && type != IGA_BASIS_BSPLINE)
    SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Basis type is %s, collocation method requires %s",IGABasisTypes[type],IGABasisTypes[IGA_BASIS_BSPLINE]);
  ierr = IGAGetBasis(iga,i,&basis);CHKERRQ(ierr);
  if (basis->type == type) PetscFunctionReturn(0);
  ierr = IGABasisSetType(basis,type);CHKERRQ(ierr);
  iga->setup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode IGASetRuleType(IGA iga,PetscInt i,IGARuleType type)
{
  IGARule        rule;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,i,2);
  PetscValidLogicalCollectiveEnum(iga,type,3);
  ierr = IGAGetRule(iga,i,&rule);CHKERRQ(ierr);
  if (rule->type == type) PetscFunctionReturn(0);
  ierr = IGARuleSetType(rule,type);CHKERRQ(ierr);
  iga->setup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode IGASetRuleSize(IGA iga,PetscInt i,PetscInt nqp)
{
  IGARule        rule;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,i,2);
  PetscValidLogicalCollectiveInt(iga,nqp,3);
  ierr = IGAGetRule(iga,i,&rule);CHKERRQ(ierr);
  if (rule->nqp == nqp) PetscFunctionReturn(0);
  ierr = IGARuleSetSize(rule,nqp);CHKERRQ(ierr);
  iga->setup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

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
  if (q <= 0) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of quadrature points %d must be positive",(int)q);
  if (rule->nqp == q) PetscFunctionReturn(0);
  ierr = IGARuleSetSize(rule,q);CHKERRQ(ierr);
  iga->setup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

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
  if (i < 0)      SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %d must be nonnegative",(int)i);
  if (i >= dim)   SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %d, but dim %d",(int)i,(int)dim);
  ierr = MPI_Comm_size(((PetscObject)iga)->comm,&size);CHKERRQ(ierr);
  if (processors < 1)
    SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of processors must be nonnegative, got %d",(int)processors);
  if (size % processors != 0)
    SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of processors %d is incompatible with communicator size %d",processors,(int)size);
  for (k=0; k<dim; k++)
    np[k] = iga->proc_sizes[k];
  np[i] = prod = processors;
  for (k=0; k<dim; k++)
    if (k!=i && np[k]>0) prod *= np[k];
  if (size % prod != 0)
    SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Processor grid sizes (%d,%d,%d) are incompatible with communicator size %d",(int)np[0],(int)np[1],(int)np[2],(int)size);
  iga->proc_sizes[i] = processors;
  PetscFunctionReturn(0);
}

PetscErrorCode IGASetUseCollocation(IGA iga,PetscBool collocation)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveBool(iga,collocation,2);
  if (iga->setupstage > 0 && iga->collocation != collocation)
    SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot change collocation after IGASetUp()");
  iga->collocation = collocation;
  PetscFunctionReturn(0);
}

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
  PetscAssertPointer(axis,3);
  dim = (iga->dim > 0) ? iga->dim : 3;
  if (i < 0)    SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %d must be nonnegative",(int)i);
  if (i >= dim) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %d, but dim %d",(int)i,(int)dim);
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
PetscErrorCode IGAGetRule(IGA iga,PetscInt i,IGARule *rule)
{
  PetscInt dim;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscAssertPointer(rule,3);
  dim = (iga->dim > 0) ? iga->dim : 3;
  if (i < 0)    SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %d must be nonnegative",(int)i);
  if (i >= dim) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %d, but dim %d",(int)i,(int)iga->dim);
  *rule = iga->rule[i];
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetBasis(IGA iga,PetscInt i,IGABasis *basis)
{
  PetscInt dim;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscAssertPointer(basis,3);
  dim = (iga->dim > 0) ? iga->dim : 3;
  if (i < 0)    SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %d must be nonnegative",(int)i);
  if (i >= dim) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %d, but dimension %d",(int)i,(int)iga->dim);
  *basis = iga->basis[i];
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetOptionsPrefix(IGA iga,const char *prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)iga,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGASetOptionsPrefix(IGA iga,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)iga,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPrependOptionsPrefix(IGA iga,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  ierr = PetscObjectPrependOptionsPrefix((PetscObject)iga,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAAppendOptionsPrefix(IGA iga,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)iga,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if PETSC_VERSION_LT(3,18,0)
#define PetscObjectProcessOptionsHandlers(obj,items) PetscObjectProcessOptionsHandlers(items,obj)
#endif

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
    PetscBool setup = iga->setup;
    PetscBool flg,flg1,flg2;
    PetscInt  i,n;
    const char *prefix = NULL;
    PetscBool collocation = iga->collocation;
    IGARuleType  rtype[3] = {IGA_RULE_LEGENDRE,IGA_RULE_LEGENDRE,IGA_RULE_LEGENDRE};
    PetscInt     rsize[3] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
    IGABasisType btype[3] = {IGA_BASIS_BSPLINE,IGA_BASIS_BSPLINE,IGA_BASIS_BSPLINE};
    PetscBool    wraps[3] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE };
    PetscInt     procs[3] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
    PetscInt     elems[3] = {16,16,16};
    PetscInt     degrs[3] = { 2, 2, 2};
    PetscInt     conts[3] = {PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE};
    PetscReal    ulims[3][2] = {{0,1},{0,1},{0,1}};
    char filename[PETSC_MAX_PATH_LEN] = {0};
    char vtype[256] = VECSTANDARD;
    char mtype[256] = MATBAIJ;
    PetscInt dim = (iga->dim > 0) ? iga->dim : 3;
    PetscInt dof = (iga->dof > 0) ? iga->dof : 1;
    PetscInt order = iga->order;

    ierr = IGAGetOptionsPrefix(iga,&prefix);CHKERRQ(ierr);

    for (i=0; i<dim; i++) {
      procs[i] = iga->proc_sizes[i];
      wraps[i] = iga->axis[i]->periodic;
      btype[i] = iga->basis[i]->type;
      rtype[i] = iga->rule[i]->type;
      if (iga->rule[i]->nqp > 0)
        rsize[i] = iga->rule[i]->nqp;
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

    PetscObjectOptionsBegin((PetscObject)iga);

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
    if (iga->collocation) {ierr = IGAOptionsReject(prefix,"-iga_rule_type");CHKERRQ(ierr);}
    if (iga->collocation) {ierr = IGAOptionsReject(prefix,"-iga_rule_size");CHKERRQ(ierr);}
    if (iga->collocation) {ierr = IGAOptionsReject(prefix,"-iga_quadrature");CHKERRQ(ierr);}

    /* Processor grid */
    ierr = PetscOptionsIntArray("-iga_processors","Processor grid","IGASetProcessors",procs,(n=dim,&n),&flg);CHKERRQ(ierr);
    if (flg) for (i=0; i<n; i++) {
        PetscInt np = procs[i];
        if (np > 0) {ierr = IGASetProcessors(iga,i,np);CHKERRQ(ierr);}
      }

    /* Periodicity */
    ierr = PetscOptionsBoolArray("-iga_periodic","Periodicity","IGAAxisSetPeriodic",wraps,(n=dim,&n),&flg);CHKERRQ(ierr);
    if (flg && n==0) for (i=0; i<dim&&i<3; i++) wraps[i] = PETSC_TRUE;
    if (flg && n==1) for (i=1; i<dim&&i<3; i++) wraps[i] = wraps[0];
    if (flg) for (i=0; i<dim; i++) {
        PetscBool w = wraps[i];
        ierr = IGAAxisSetPeriodic(iga->axis[i],w);CHKERRQ(ierr);
      }

    /* Geometry */
    ierr = PetscOptionsString("-iga_load","Specify IGA geometry file","IGARead",filename,filename,sizeof(filename),&flg);CHKERRQ(ierr);
    if (!flg) {ierr = PetscOptionsString("-iga_geometry","deprecated, use -iga_load","IGARead",filename,filename,sizeof(filename),&flg);CHKERRQ(ierr);}
    if (flg) { /* load from file */
      ierr = IGAOptionsReject(prefix,"-iga_elements");CHKERRQ(ierr);
      ierr = IGAOptionsReject(prefix,"-iga_degree");CHKERRQ(ierr);
      ierr = IGAOptionsReject(prefix,"-iga_continuity");CHKERRQ(ierr);
      ierr = IGAOptionsReject(prefix,"-iga_limits");CHKERRQ(ierr);
      ierr = IGARead(iga,filename);CHKERRQ(ierr);
      ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    } else { /* set axis details */
      PetscInt ne,nd,nc,nl;
      ierr = PetscOptionsEnumArray("-iga_basis_type","Basis type","IGASetBasisType",IGABasisTypes,(PetscEnum*)btype,(n=dim,&n),&flg);CHKERRQ(ierr);
      if (flg && n==1) for (i=1; i<dim&&i<3; i++) btype[i] = btype[0];
      for (i=0; i<dim&&i<3; i++) if (btype[i] != IGA_BASIS_BSPLINE) conts[i] = 0;
      ierr = PetscOptionsIntArray ("-iga_elements",  "Elements",  "IGAAxisInitUniform",elems,(ne=dim,&ne),&flg1);CHKERRQ(ierr);
      ierr = PetscOptionsIntArray ("-iga_degree",    "Degree",    "IGAAxisSetDegree",  degrs,(nd=dim,&nd),&flg2);CHKERRQ(ierr);
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
        if ((flg1||flg2) || (axis->p==0||axis->m==1)) {
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
    order = PetscClipInterval(order,1,4);
    ierr = PetscOptionsInt("-iga_order","Maximum available derivative order","IGASetOrder",order,&order,&flg);CHKERRQ(ierr);
    if (flg) { ierr = IGASetOrder(iga,order);CHKERRQ(ierr);}

    /* Basis Type */
    ierr = PetscOptionsEnumArray("-iga_basis_type","Basis type","IGASetBasisType",IGABasisTypes,(PetscEnum*)btype,(n=dim,&n),&flg);CHKERRQ(ierr);
    if (flg && n==1) for (i=1; i<dim&&i<3; i++) btype[i] = btype[0];
    if (flg) for (i=0; i<dim; i++) {
        ierr = IGASetBasisType(iga,i,btype[i]);CHKERRQ(ierr);
      }

    /* Quadrature Rule type & size */
    for (i=0; i<dim&&i<3; i++) if (rsize[i] < 1) rsize[i] = iga->axis[i]->p + 1;
    ierr = PetscOptionsEnumArray("-iga_rule_type","Quadrature Rule type","IGASetRuleType",IGARuleTypes,(PetscEnum*)rtype,(n=dim,&n),&flg1);CHKERRQ(ierr);
    if (flg1 && n==1) for (i=1; i<dim&&i<3; i++) rtype[i] = rtype[0];
    ierr = PetscOptionsIntArray("-iga_rule_size","Quadrature Rule size","IGASetRuleSize",rsize,(n=dim,&n),&flg2);CHKERRQ(ierr);
    if (flg2 && n==1) for (i=1; i<dim&&i<3; i++) rsize[i] = rsize[0];
    if (flg1 || flg2) for (i=0; i<dim&&i<3; i++) {
        PetscInt nqp = (rsize[i] > 0) ? rsize[i] : (iga->rule[i]->nqp > 0) ? iga->rule[i]->nqp : iga->axis[i]->p + 1;
        if (flg1 && flg2) {ierr = IGARuleReset(iga->rule[i]);CHKERRQ(ierr);}
        if (flg1) {ierr = IGASetRuleType(iga,i,rtype[i]);CHKERRQ(ierr);}
        if (flg2) {ierr = IGASetRuleSize(iga,i,nqp);CHKERRQ(ierr);}
      }
    /* Quadrature (Legacy option)*/
    for (i=0; i<dim&&i<3; i++) if (rsize[i] < 1) rsize[i] = iga->axis[i]->p + 1;
    ierr = PetscOptionsIntArray("-iga_quadrature","Quadrature points","IGASetQuadrature",rsize,(n=dim,&n),&flg);CHKERRQ(ierr);
    if (flg) for (i=0; i<dim; i++) {
        PetscInt q = (i<n) ? rsize[i] : rsize[0];
        if (q > 0) {ierr = IGASetQuadrature(iga,i,q);CHKERRQ(ierr);}
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
    ierr = PetscOptionsName("-iga_view_draw",  "Draw to screen",                  "IGAView",NULL);CHKERRQ(ierr);

    ierr = PetscObjectProcessOptionsHandlers((PetscObject)iga,PetscOptionsObject);CHKERRQ(ierr);
    PetscOptionsEnd();

    if (setup) {ierr = IGASetUp(iga);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode IGACreateSubComms1D(IGA,MPI_Comm[]);

PetscErrorCode IGACreateSubComms1D(IGA iga,MPI_Comm subcomms[])
{
  MPI_Comm       comm;
  PetscInt       i,dim;
  PetscMPIInt    size;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscAssertPointer(subcomms,2);
  IGACheckSetUpStage1(iga,1);

  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1 || dim == 1)
    for (i=0; i<dim; i++)
      {ierr = MPI_Comm_dup(comm,&subcomms[i]);CHKERRQ(ierr);}
#if !defined(PETSC_HAVE_MPIUNI)
  else {
    MPI_Comm    cartcomm;
    PetscMPIInt ndims,dims[3],periods[3]={0,0,0},reorder=0;
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

PetscErrorCode IGACreateDMDA(IGA iga,
                             PetscInt bs,
                             const PetscInt  gsizes[],
                             const PetscInt  lsizes[],
                             const PetscBool periodic[],
                             PetscBool stencil_box,
                             PetscInt  stencil_width,
                             DM *dm)
{
  PetscInt        i,dim;
  MPI_Comm        subcomms[3];
  PetscInt        procs[3]   = {-1,-1,-1};
  PetscInt        sizes[3]   = { 1, 1, 1};
  PetscInt        width[3]   = { 1, 1, 1};
  PetscInt        *ranges[3] = {NULL, NULL, NULL};
  DMBoundaryType  btype[3]   = {DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE};
  DMDAStencilType stype      = stencil_box ? DMDA_STENCIL_BOX : DMDA_STENCIL_STAR;
  PetscInt        swidth     = stencil_width;
  DM              da;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  PetscAssertPointer(gsizes,3);
  PetscAssertPointer(lsizes,4);
  if (periodic) PetscAssertPointer(periodic,5);
  PetscValidLogicalCollectiveBool(iga,stencil_box,6);
  PetscValidLogicalCollectiveInt(iga,stencil_width,7);
  PetscAssertPointer(dm,8);
  IGACheckSetUpStage1(iga,1);

  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  for (i=0; i<dim; i++) {
    procs[i] = iga->proc_sizes[i];
    btype[i] = (periodic && periodic[i]) ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE;
    sizes[i] = gsizes[i];
    width[i] = lsizes[i];
  }
  ierr = IGACreateSubComms1D(iga,subcomms);CHKERRQ(ierr);
  for (i=0; i<dim; i++) {
    ierr = PetscMalloc1((size_t)procs[i],&ranges[i]);CHKERRQ(ierr);
    ierr = MPI_Allgather(&width[i],1,MPIU_INT,ranges[i],1,MPIU_INT,subcomms[i]);CHKERRQ(ierr);
    ierr = MPI_Comm_free(&subcomms[i]);CHKERRQ(ierr);
  }
  ierr = DMDACreate(((PetscObject)iga)->comm,&da);CHKERRQ(ierr);
  ierr = DMSetDimension(da,dim);CHKERRQ(ierr);
  ierr = DMDASetDof(da,bs);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(da,procs[0],procs[1],procs[2]);CHKERRQ(ierr);
  ierr = DMDASetSizes(da,sizes[0],sizes[1],sizes[2]);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(da,ranges[0],ranges[1],ranges[2]);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(da,btype[0],btype[1],btype[2]);CHKERRQ(ierr);
  ierr = DMDASetStencilType(da,stype);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(da,swidth);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  for (i=0; i<dim; i++) {ierr = PetscFree(ranges[i]);CHKERRQ(ierr);}

  *dm = da;
  PetscFunctionReturn(0);
}

PetscErrorCode IGACreateElemDM(IGA iga,PetscInt bs,DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  PetscAssertPointer(dm,3);
  IGACheckSetUpStage1(iga,1);
  ierr = IGACreateDMDA(iga,bs,iga->elem_sizes,iga->elem_width,
                       NULL,PETSC_TRUE,0,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGACreateGeomDM(IGA iga,PetscInt bs,DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  PetscAssertPointer(dm,3);
  IGACheckSetUpStage1(iga,1);
  ierr = IGACreateDMDA(iga,bs,iga->geom_sizes,iga->geom_lwidth,
                       NULL,PETSC_TRUE,0,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscAssertPointer(dm,3);
  IGACheckSetUpStage1(iga,1);
  ierr = IGACreateDMDA(iga,bs,iga->node_sizes,iga->node_lwidth,
                       NULL,PETSC_TRUE,0,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetElemDM(IGA iga,DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscAssertPointer(dm,3);
  IGACheckSetUpStage2(iga,1);
  if (!iga->elem_dm) {ierr = IGACreateElemDM(iga,iga->dof,&iga->elem_dm);CHKERRQ(ierr);}
  *dm = iga->elem_dm;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetGeomDM(IGA iga,DM *dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscAssertPointer(dm,3);
  IGACheckSetUpStage2(iga,1);
  if (!iga->geom_dm) {ierr = IGACreateGeomDM(iga,iga->dof,&iga->geom_dm);CHKERRQ(ierr);}
  *dm = iga->geom_dm;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetNodeDM(IGA iga,DM *dm)
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscAssertPointer(dm,3);
  IGACheckSetUpStage2(iga,1);
  if (!iga->node_dm) {
    ierr = IGACreateNodeDM(iga,iga->dof,&iga->node_dm);CHKERRQ(ierr);
    if (iga->fieldname)
      for (i=0; i<iga->dof; i++)
        {ierr = DMDASetFieldName(iga->node_dm,i,iga->fieldname[i]);CHKERRQ(ierr);}
  }
  *dm = iga->node_dm;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode IGA_Partition(PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt[],PetscInt[]);
PETSC_EXTERN PetscErrorCode IGA_Distribute(PetscInt,const PetscInt[],const PetscInt[],const PetscInt[],PetscInt[],PetscInt[]);

EXTERN_C_BEGIN
extern PetscReal IGA_Greville(PetscInt i,PetscInt p,const PetscReal U[]);
extern PetscInt  IGA_FindSpan(PetscInt n,PetscInt p,PetscReal u, const PetscReal U[]);
EXTERN_C_END

PETSC_EXTERN PetscErrorCode IGASetUp_Stage1(IGA);
PetscErrorCode IGASetUp_Stage1(IGA iga)
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

  if (!iga->collocation) {
    for (i=0; i<dim; i++) grid_sizes[i] = iga->axis[i]->nel;
  } else {
    for (i=0; i<dim; i++) grid_sizes[i] = iga->axis[i]->nnp;
  }

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
  }

  if (!iga->collocation) {

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
        if (elast < nel-1) {
          lend = span[elast+1] - p;
        } else {
          lend = span[elast] + 1;
        }
        geom_sizes[i]  = size;
        geom_lstart[i] = lstart;
        geom_lwidth[i] = lend - lstart;
        geom_gstart[i] = gstart;
        geom_gwidth[i] = gend - gstart;
      }
    }
    { /* node partitioning */
      PetscInt *node_shift  = iga->node_shift;
      PetscInt *node_sizes  = iga->node_sizes;
      PetscInt *node_lstart = iga->node_lstart;
      PetscInt *node_lwidth = iga->node_lwidth;
      PetscInt *node_gstart = iga->node_gstart;
      PetscInt *node_gwidth = iga->node_gwidth;
      for (i=0; i<dim; i++) {
        PetscInt size = iga->proc_sizes[i];
        PetscInt rank = iga->proc_ranks[i];
        PetscInt nnp  = iga->axis[i]->nnp;
        node_shift[i]  = 0;
        node_sizes[i]  = nnp;
        node_lstart[i] = iga->geom_lstart[i];
        node_lwidth[i] = iga->geom_lwidth[i];
        node_gstart[i] = iga->geom_gstart[i];
        node_gwidth[i] = iga->geom_gwidth[i];
        if (rank == size-1)
          node_lwidth[i] = node_sizes[i] - node_lstart[i];
      }
    }

  } else {

    { /* node partitioning */
      PetscInt *node_shift  = iga->node_shift;
      PetscInt *node_sizes  = iga->node_sizes;
      PetscInt *node_lstart = iga->node_lstart;
      PetscInt *node_lwidth = iga->node_lwidth;
      PetscInt *node_gstart = iga->node_gstart;
      PetscInt *node_gwidth = iga->node_gwidth;
      for (i=0; i<dim; i++) {
        node_sizes[i]  = iga->elem_sizes[i];
        node_lstart[i] = iga->elem_start[i];
        node_lwidth[i] = iga->elem_width[i];
      }
      for (i=0; i<dim; i++) {
        PetscInt   p = iga->axis[i]->p;
        PetscInt   m = iga->axis[i]->m;
        PetscReal *U = iga->axis[i]->U;
        PetscInt   n = m - p - 1;
        PetscInt   shift = (n + 1 - node_sizes[i])/2;
        node_shift[i] = shift;
        {
          PetscInt  a = node_lstart[i];
          PetscReal u = IGA_Greville(a+shift,p,U);
          PetscInt  k = IGA_FindSpan(n,p,u,U)-shift;
          node_gstart[i] = k - p;
        }
        {
          PetscInt  a = node_lstart[i] + node_lwidth[i] - 1;
          PetscReal u = IGA_Greville(a+shift,p,U);
          PetscInt  k = IGA_FindSpan(n,p,u,U)-shift;
          node_gwidth[i] = k + 1 - node_gstart[i];
        }
      }
    }
    { /* geometry partitioning */
      PetscInt *geom_sizes  = iga->geom_sizes;
      PetscInt *geom_lstart = iga->geom_lstart;
      PetscInt *geom_lwidth = iga->geom_lwidth;
      PetscInt *geom_gstart = iga->geom_gstart;
      PetscInt *geom_gwidth = iga->geom_gwidth;
      for (i=0; i<dim; i++) {
        PetscInt size = iga->proc_sizes[i];
        PetscInt rank = iga->proc_ranks[i];
        PetscInt p = iga->axis[i]->p;
        PetscInt m = iga->axis[i]->m;
        PetscInt n = m - p - 1;
        PetscInt shift = iga->node_shift[i];
        geom_sizes[i]  = n + 1;
        geom_lstart[i] = iga->node_lstart[i] + shift;
        geom_lwidth[i] = iga->node_lwidth[i];
        geom_gstart[i] = iga->node_gstart[i] + shift;
        geom_gwidth[i] = iga->node_gwidth[i];
        if (rank == 0) {
          geom_lstart[i] -= shift;
          geom_lwidth[i] += shift;
        }
        if (rank == size-1) {
          geom_lwidth[i] = geom_sizes[i] - geom_lstart[i];
          geom_gwidth[i] = geom_sizes[i] - geom_gstart[i];
        }
      }
    }

  }

  for (i=dim; i<3; i++) {
    iga->elem_sizes[i]  = 1;
    iga->elem_start[i]  = 0;
    iga->elem_width[i]  = 1;
    iga->geom_sizes[i]  = 1;
    iga->geom_lstart[i] = 0;
    iga->geom_lwidth[i] = 1;
    iga->geom_gstart[i] = 0;
    iga->geom_gwidth[i] = 1;
    iga->node_shift[i]  = 0;
    iga->node_sizes[i]  = 1;
    iga->node_lstart[i] = 0;
    iga->node_lwidth[i] = 1;
    iga->node_gstart[i] = 0;
    iga->node_gwidth[i] = 1;
  }

  iga->rational = PETSC_FALSE;
  iga->geometry = 0;
  iga->property = 0;
  ierr = PetscFree(iga->rationalW);CHKERRQ(ierr);
  ierr = PetscFree(iga->geometryX);CHKERRQ(ierr);
  ierr = PetscFree(iga->propertyA);CHKERRQ(ierr);

  iga->fixtable = PETSC_FALSE;
  ierr = PetscFree(iga->fixtableU);CHKERRQ(ierr);

  ierr = DMDestroy(&iga->elem_dm);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->geom_dm);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->node_dm);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->draw_dm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode IGASetUp_Stage2(IGA);
PetscErrorCode IGASetUp_Stage2(IGA iga)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (iga->setupstage >= 2) PetscFunctionReturn(0);
  iga->setupstage = 2;

  if (iga->dof < 1) iga->dof = 1;  /* XXX Error ? */

  if (!iga->vectype) {
    const MatType vtype = VECSTANDARD;
    ierr = IGASetVecType(iga,vtype);CHKERRQ(ierr);
  }
  if (!iga->mattype) {
    const MatType mtype = (iga->dof > 1) ? MATBAIJ : MATAIJ;
    ierr = IGASetMatType(iga,mtype);CHKERRQ(ierr);
  }

  ierr = AODestroy(&iga->ao);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&iga->lgmap);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&iga->map);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->g2l);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->l2g);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->l2l);CHKERRQ(ierr);
  ierr = VecDestroy(&iga->natural);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->n2g);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&iga->g2n);CHKERRQ(ierr);
  while (iga->nwork > 0)
    {ierr = VecDestroy(&iga->vwork[--iga->nwork]);CHKERRQ(ierr);}
  {
    IGA_Grid grid;
    /* create the grid context */
    ierr = IGA_Grid_Create(((PetscObject)iga)->comm,&grid);CHKERRQ(ierr);
    ierr = IGA_Grid_Init(grid,
                         iga->dim,iga->dof,iga->node_sizes,
                         iga->node_lstart,iga->node_lwidth,
                         iga->node_gstart,iga->node_gwidth);CHKERRQ(ierr);
    /* build the application ordering */
    ierr = IGA_Grid_GetAO(grid,&iga->ao);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->ao);CHKERRQ(ierr);
    /* build the local to global mapping */
    ierr = IGA_Grid_GetLGMap(grid,&iga->lgmap);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->lgmap);CHKERRQ(ierr);
    /* build the layout */
    ierr = IGA_Grid_GetLayout(grid,&iga->map);CHKERRQ(ierr);
    ierr = PetscLayoutReference(iga->map,&iga->map);CHKERRQ(ierr);
    /* build global <-> local vector scatters */
    ierr = IGA_Grid_GetScatterG2L(grid,&iga->g2l);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->g2l);CHKERRQ(ierr);
    ierr = IGA_Grid_GetScatterL2G(grid,&iga->l2g);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)iga->l2g);CHKERRQ(ierr);
    /* build global <-> natural vector scatter */
    ierr = IGA_Grid_NewScatterApp(grid,iga->node_shift,
                                  iga->geom_sizes,iga->geom_lstart,iga->geom_lwidth,
                                  &iga->natural,&iga->n2g,&iga->g2n);CHKERRQ(ierr);
    /* destroy the grid context */
    ierr = IGA_Grid_Destroy(&grid);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode IGASetUp_Basic(IGA);

PetscErrorCode IGASetUp_Basic(IGA iga)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (iga->dim < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE, "Must call IGASetDim() first");
  ierr = IGASetUp_Stage1(iga);;CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IGASetUp_View(IGA iga)
{
  PetscBool      ascii = PETSC_FALSE;
  PetscBool      info = PETSC_FALSE;
  PetscBool      binary = PETSC_FALSE;
  PetscBool      draw = PETSC_FALSE;
  char           filename1[PETSC_MAX_PATH_LEN] = "";
  char           filename2[PETSC_MAX_PATH_LEN] = "";
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);

  PetscObjectOptionsBegin((PetscObject)iga);
  ierr = PetscOptionsString("-iga_view_ascii",  "Information on IGA context",      "IGAView",filename1,filename1,PETSC_MAX_PATH_LEN,&ascii);CHKERRQ(ierr);
  ierr = PetscOptionsBool  ("-iga_view_info",   "Output more detailed information","IGAView",info,&info,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool  ("-iga_view_detail", "Output more detailed information","IGAView",info,&info,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-iga_view_binary", "Save to file in binary format",   "IGAView",filename2,filename2,PETSC_MAX_PATH_LEN,&binary);CHKERRQ(ierr);
  ierr = PetscOptionsBool  ("-iga_view_draw",   "Draw to screen",                  "IGAView",draw,&draw,NULL);CHKERRQ(ierr);
  PetscOptionsEnd();

  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  if ((ascii||info) && !PetscPreLoadingOn) {
    ierr = PetscViewerASCIIOpen(comm,filename1,&viewer);CHKERRQ(ierr);
    if (info) {ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);}
    ierr = IGAPrint(iga,viewer);CHKERRQ(ierr);
    if (info) {ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);}
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  if (binary && !PetscPreLoadingOn) {
    if (filename2[0]) {
      ierr = IGAWrite(iga,filename2);CHKERRQ(ierr);
    } else {
      viewer = PETSC_VIEWER_BINARY_(comm);
      ierr = IGASave(iga,viewer);CHKERRQ(ierr);
    }
  }
  if (draw && !PetscPreLoadingOn) {
    int h = 600, w = 600;
    ierr = PetscViewerDrawOpen(comm,NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,w,h,&viewer);CHKERRQ(ierr);
    ierr = IGADraw(iga,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


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
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetDim() first");

  iga->setup = PETSC_TRUE;

  /* --- Stage 1 --- */
  ierr = IGASetUp_Stage1(iga);CHKERRQ(ierr);

  /* --- Stage 2 --- */
  ierr = IGASetUp_Stage2(iga);CHKERRQ(ierr);

  /* --- Stage 3 --- */
  iga->setupstage = 3;

  if (iga->order < 0) {
    for (i=0; i<iga->dim; i++)
      iga->order = PetscMax(iga->order,iga->axis[i]->p);
    ierr = IGASetOrder(iga,iga->order);CHKERRQ(ierr);
  }

  for (i=0; i<3; i++)
    if (!iga->collocation) {
      if (i >= iga->dim) {ierr = IGARuleReset(iga->rule[i]);CHKERRQ(ierr);}
      if (i >= iga->dim) {ierr = IGARuleSetType(iga->rule[i],IGA_RULE_LEGENDRE);CHKERRQ(ierr);}
      ierr = IGABasisInitQuadrature(iga->basis[i],iga->axis[i],iga->rule[i]);CHKERRQ(ierr);
    } else {
      ierr = IGARuleReset(iga->rule[i]);CHKERRQ(ierr);
      ierr = IGABasisInitCollocation(iga->basis[i],iga->axis[i]);CHKERRQ(ierr);
    }

  ierr = IGAElementInit(iga->iterator,iga);CHKERRQ(ierr);

  ierr = IGAViewFromOptions(iga,NULL,"-iga_view");CHKERRQ(ierr);
  ierr = IGASetUp_View(iga);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAClone(IGA iga,PetscInt dof,IGA *_newiga)
{
  MPI_Comm       comm;
  IGA            newiga;
  PetscInt       i,n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,dof,2);
  PetscAssertPointer(_newiga,3);
  IGACheckSetUp(iga,1);

  *_newiga = NULL;
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = IGACreate(comm,&newiga);CHKERRQ(ierr);
  *_newiga = newiga;

  newiga->collocation = iga->collocation;

  newiga->dim = iga->dim;
  for (i=0; i<3; i++) {
    ierr = IGAAxisCopy(iga->axis[i],newiga->axis[i]);CHKERRQ(ierr);
    ierr = IGARuleCopy(iga->rule[i],newiga->rule[i]);CHKERRQ(ierr);
    newiga->basis[i]->type = iga->basis[i]->type;
    newiga->proc_sizes[i]  = iga->proc_sizes[i];
    newiga->proc_ranks[i]  = iga->proc_ranks[i];
    newiga->elem_sizes[i]  = iga->elem_sizes[i];
    newiga->elem_start[i]  = iga->elem_start[i];
    newiga->elem_width[i]  = iga->elem_width[i];
    newiga->geom_sizes[i]  = iga->geom_sizes[i];
    newiga->geom_lstart[i] = iga->geom_lstart[i];
    newiga->geom_lwidth[i] = iga->geom_lwidth[i];
    newiga->geom_gstart[i] = iga->geom_gstart[i];
    newiga->geom_gwidth[i] = iga->geom_gwidth[i];
    newiga->node_shift[i]  = iga->node_shift[i];
    newiga->node_sizes[i]  = iga->node_sizes[i];
    newiga->node_lstart[i] = iga->node_lstart[i];
    newiga->node_lwidth[i] = iga->node_lwidth[i];
    newiga->node_gstart[i] = iga->node_gstart[i];
    newiga->node_gwidth[i] = iga->node_gwidth[i];
  }
  newiga->setupstage = 1;

  n  = iga->geom_gwidth[0];
  n *= iga->geom_gwidth[1];
  n *= iga->geom_gwidth[2];
  if (iga->rational && iga->rationalW) {
    newiga->rational = iga->rational;
    ierr = PetscMalloc1((size_t)n,&newiga->rationalW);CHKERRQ(ierr);
    ierr = PetscMemcpy(newiga->rationalW,iga->rationalW,(size_t)n*sizeof(PetscReal));CHKERRQ(ierr);
  }
  if (iga->geometry && iga->geometryX) {
    PetscInt nsd = newiga->geometry = iga->geometry;
    ierr = PetscMalloc1((size_t)(n*nsd),&newiga->geometryX);CHKERRQ(ierr);
    ierr = PetscMemcpy(newiga->geometryX,iga->geometryX,(size_t)(n*nsd)*sizeof(PetscReal));CHKERRQ(ierr);
  }
  if (iga->property && iga->propertyA) {
    PetscInt npd = newiga->property = iga->property;
    ierr = PetscMalloc1((size_t)(n*npd),&newiga->propertyA);CHKERRQ(ierr);
    ierr = PetscMemcpy(newiga->propertyA,iga->propertyA,(size_t)(n*npd)*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  newiga->dof = (dof > 0) ? dof : iga->dof;
  ierr = IGASetUp_Stage2(newiga);CHKERRQ(ierr);

  newiga->order = iga->order;
  ierr = IGASetUp(newiga);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode IGASetVecType(IGA iga,const VecType vectype)
{
  char           *vtype;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscAssertPointer(vectype,2);
  ierr = PetscStrallocpy(vectype,&vtype);CHKERRQ(ierr);
  ierr = PetscFree(iga->vectype);CHKERRQ(ierr);
  iga->vectype = vtype;
  PetscFunctionReturn(0);
}

PetscErrorCode IGASetMatType(IGA iga,const MatType mattype)
{
  char           *mtype;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscAssertPointer(mattype,2);
  ierr = PetscStrallocpy(mattype,&mtype);CHKERRQ(ierr);
  ierr = PetscFree(iga->mattype);CHKERRQ(ierr);
  iga->mattype = mtype;
  PetscFunctionReturn(0);
}
