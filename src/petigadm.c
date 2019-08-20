#include <petiga.h>
#include "petsc/private/dmimpl.h"

typedef struct {
  IGA iga;
} DM_IGA;

#define DMIGACast(dm) ((DM_IGA*)(dm)->data)

PetscErrorCode DMIGACreate(IGA iga,DM *dm)
{
  MPI_Comm       comm;
  const char     *prefix;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(dm,2);
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = IGAGetOptionsPrefix(iga,&prefix);CHKERRQ(ierr);
  ierr = DMCreate(comm,dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm,DMIGA);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(*dm,prefix);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)iga);CHKERRQ(ierr);
  DMIGACast(*dm)->iga = iga;
  if (iga->setup) {ierr = DMSetUp(*dm);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMIGASetIGA(DM dm,IGA iga)
{
  IGA            dmiga;
  PetscBool      match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMIGA,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONG,"DM is not of type DMIGA");
  dmiga = DMIGACast(dm)->iga;
  if (dmiga && dmiga != iga) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONGSTATE,"IGA already set in DMIGA");
  ierr = PetscObjectReference((PetscObject)iga);CHKERRQ(ierr);
  ierr = IGADestroy(&dmiga);CHKERRQ(ierr);
  DMIGACast(dm)->iga = iga;
  PetscFunctionReturn(0);
}

PetscErrorCode DMIGAGetIGA(DM dm,IGA *iga)
{
  PetscBool      match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(iga,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMIGA,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_ARG_WRONG,"DM is not of type DMIGA");
  *iga = DMIGACast(dm)->iga;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMIGA_GetIGA(DM dm,IGA *iga)
{
  IGA            dmiga = DMIGACast(dm)->iga;
  MPI_Comm       comm;
  const char     *prefix;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!dmiga) {
    ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
    ierr = PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix);CHKERRQ(ierr);
    ierr = IGACreate(comm,&dmiga);CHKERRQ(ierr);
    ierr = IGASetOptionsPrefix(dmiga,prefix);CHKERRQ(ierr);
    ierr = IGASetFromOptions(dmiga);CHKERRQ(ierr);
    DMIGACast(dm)->iga = dmiga;
  }
  *iga = dmiga;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGetLocalToGlobalMapping_IGA(DM dm)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  IGACheckSetUpStage2(iga,0);
  ierr = PetscObjectReference((PetscObject)iga->lgmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&dm->ltogmap);CHKERRQ(ierr);
  dm->ltogmap = iga->lgmap;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGlobalToLocalBegin_IGA(DM dm,Vec g,InsertMode mode,Vec l)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAGlobalToLocalBegin(iga,g,l,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode DMGlobalToLocalEnd_IGA(DM dm,Vec g,InsertMode mode,Vec l)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAGlobalToLocalEnd(iga,g,l,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode DMLocalToGlobalBegin_IGA(DM dm,Vec l,InsertMode mode,Vec g)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGALocalToGlobalBegin(iga,l,g,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode DMLocalToGlobalEnd_IGA(DM dm,Vec l,InsertMode mode,Vec g)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGALocalToGlobalEnd(iga,l,g,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode DMLocalToLocalBegin_IGA(DM dm,Vec g,InsertMode mode,Vec l)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGALocalToLocalBegin(iga,g,l,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode DMLocalToLocalEnd_IGA(DM dm,Vec g,InsertMode mode,Vec l)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGALocalToLocalEnd(iga,g,l,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateGlobalVector_IGA(DM dm,Vec *gvec)
{
  IGA            iga = DMIGACast(dm)->iga;
  VecType        vtype,save;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  vtype = dm->vectype;
  save = iga->vectype;
  if (vtype) iga->vectype = (char*)vtype;
  ierr = IGACreateVec(iga,gvec);CHKERRQ(ierr);
  if (vtype) iga->vectype = (char*)save;
  ierr = VecSetDM(*gvec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateLocalVector_IGA(DM dm,Vec *lvec)
{
  IGA            iga = DMIGACast(dm)->iga;
  VecType        vtype,save;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  vtype = dm->vectype;
  save = iga->vectype;
  if (vtype) iga->vectype = (char*)vtype;
  ierr = IGACreateLocalVec(iga,lvec);CHKERRQ(ierr);
  if (vtype) iga->vectype = (char*)save;
  ierr = VecSetDM(*lvec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateMatrix_IGA(DM dm,Mat *J)
{
  IGA            iga = DMIGACast(dm)->iga;
  MatType        mtype = dm->mattype;
  MatType        save;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  save = iga->mattype;
  if (mtype) iga->mattype = (char*)mtype;
  ierr = IGACreateMat(iga,J);CHKERRQ(ierr);
  if (mtype) iga->mattype = (char*)save;
  ierr = MatSetDM(*J,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDestroy_IGA(DM dm)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFree(dm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetFromOptions_IGA(PETSC_UNUSED PetscOptionItems *PetscOptionsObject,DM dm)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!iga) PetscFunctionReturn(0);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetUp_IGA(DM dm)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscInt       dim,nsd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMIGA_GetIGA(dm,&iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping_IGA(dm);CHKERRQ(ierr);
  ierr = IGAGetDof(iga,&dm->bs);CHKERRQ(ierr);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = IGAGetGeometryDim(iga,&nsd);CHKERRQ(ierr);
  ierr = DMSetDimension(dm,dim);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(dm,nsd>=1?nsd:dim);CHKERRQ(ierr);
  ierr = DMSetVecType(dm,iga->vectype);CHKERRQ(ierr);
  ierr = DMSetMatType(dm,iga->mattype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMView_IGA(DM dm,PetscViewer viewer)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!iga) PetscFunctionReturn(0);
  ierr = IGAView(iga,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLoad_IGA(DM dm,PetscViewer viewer)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMIGA_GetIGA(dm,&iga);CHKERRQ(ierr);
  ierr = IGALoad(iga,viewer);CHKERRQ(ierr);
  if (dm->setupcalled) {
    ierr = DMSetUp_IGA(dm);CHKERRQ(ierr);
  } else {
    ierr = DMSetUp(dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMClone_IGA(DM dm,DM *newdm)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscInt       dof;
  IGA            newiga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);
  ierr = IGAClone(iga,dof,&newiga);CHKERRQ(ierr);
  ierr = DMSetType(*newdm,DMIGA);CHKERRQ(ierr);
  ierr = DMIGASetIGA(*newdm,newiga);CHKERRQ(ierr);
  ierr = IGADestroy(&newiga);CHKERRQ(ierr);
  ierr = DMSetUp(*newdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateCoordinateDM_IGA(DM dm,DM *cdm)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscInt       dim,nsd;
  IGA            ciga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = IGAGetGeometryDim(iga,&nsd);CHKERRQ(ierr);
  ierr = IGAClone(iga,PetscMax(dim,nsd),&ciga);CHKERRQ(ierr);
  ierr = DMIGACreate(ciga,cdm);CHKERRQ(ierr);
  ierr = IGADestroy(&ciga);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if PETSC_VERSION_LT(3,9,0)
static PetscErrorCode DMCreateSubDM_IGA(DM dm,PetscInt numFields,PetscInt fields[],IS *is,DM *subdm)
#else
static PetscErrorCode DMCreateSubDM_IGA(DM dm,PetscInt numFields,const PetscInt fields[],IS *is,DM *subdm)
#endif
{
  IGA            iga = DMIGACast(dm)->iga;
  IGA            subiga;
  PetscInt       i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(dm,numFields,2);
  for (i=0; i<numFields; i++)
    PetscValidLogicalCollectiveInt(dm,fields[i],3);
  for (i=0; i<numFields; i++)
    if (fields[i] < 0 || fields[i] >= iga->dof)
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
               "Field number %d must be in range [0,%D], got %D",i,iga->dof-1,fields[i]);
  if (is) {
    MPI_Comm comm;
    PetscInt n,bs,start,end;
    PetscInt j,count = 0,*indices = NULL;
    ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
    ierr = PetscLayoutGetBlockSize(iga->map,&bs);CHKERRQ(ierr);
    ierr = PetscLayoutGetRange(iga->map,&start,&end);CHKERRQ(ierr);
    start /= bs; end /= bs; n = end-start;
    ierr = PetscMalloc1((size_t)(n*numFields),&indices);CHKERRQ(ierr);
    for (j=start; j<end; j++)
      for (i=0; i<numFields; i++)
        indices[count++] = j*bs + fields[i];
    ierr = ISCreateGeneral(comm,count,indices,PETSC_OWN_POINTER,is);CHKERRQ(ierr);
  }
  if (subdm) {
    ierr = IGAClone(iga,numFields,&subiga);CHKERRQ(ierr);
    if (iga->fieldname) {
      for (i=0; i<numFields; i++)
        {ierr = IGASetFieldName(subiga,i,iga->fieldname[fields[i]]);CHKERRQ(ierr);}
    }
    ierr = DMIGACreate(subiga,subdm);CHKERRQ(ierr);
    ierr = IGADestroy(&subiga);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateFieldIS_IGA(DM dm,PetscInt *numFields,char ***fieldNames,IS **fields)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscInt       i,dof;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);
  if (numFields) *numFields = dof;
  if (fieldNames) {
    ierr = PetscMalloc1((size_t)dof,fieldNames);CHKERRQ(ierr);
    for (i=0; i<dof; i++) {
      const char *fieldname; char buf[256];
      ierr = IGAGetFieldName(iga,i,&fieldname);CHKERRQ(ierr);
      if (!fieldname) {
        ierr = PetscSNPrintf(buf,sizeof(buf),"%D",i);CHKERRQ(ierr);
        fieldname = buf;
      }
      ierr = PetscStrallocpy(fieldname,&(*fieldNames)[i]);CHKERRQ(ierr);
    }
  }
  if (fields) {
    MPI_Comm comm;
    PetscInt n,bs,start,end;
    ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
    ierr = PetscLayoutGetBlockSize(iga->map,&bs);CHKERRQ(ierr);
    ierr = PetscLayoutGetRange(iga->map,&start,&end);CHKERRQ(ierr);
    start /= bs; end /= bs; n = end-start;
    ierr = PetscMalloc1((size_t)dof,fields);CHKERRQ(ierr);
    for (i=0; i<bs; i++) {ierr = ISCreateStride(comm,n,start+i,bs,&(*fields)[i]);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateFieldDecomposition_IGA(DM dm,PetscInt *len,char ***namelist,IS **islist,DM **dmlist)
{
  PetscInt       i,numFields;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMCreateFieldIS_IGA(dm,&numFields,namelist,islist);CHKERRQ(ierr);
  if (len) *len = numFields;
  if (dmlist) {
    ierr = PetscMalloc1((size_t)numFields,dmlist);CHKERRQ(ierr);
    for (i=0; i<numFields; i++) {
      PetscInt *fields = &i;
      ierr = DMCreateSubDM_IGA(dm,1,fields,NULL,&(*dmlist)[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
PetscErrorCode DMCreate_IGA(DM dm);
PetscErrorCode DMCreate_IGA(DM dm)
{
  DM_IGA         *dd = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(dm,1);

  ierr = PetscNewLog(dm,&dd);CHKERRQ(ierr);
  dm->data = dd;

  dm->ops->destroy                      = DMDestroy_IGA;
  dm->ops->view                         = DMView_IGA;
  dm->ops->setfromoptions               = DMSetFromOptions_IGA;
  dm->ops->setup                        = DMSetUp_IGA;
  dm->ops->load                         = DMLoad_IGA;

  dm->ops->createglobalvector           = DMCreateGlobalVector_IGA;
  dm->ops->createlocalvector            = DMCreateLocalVector_IGA;
  dm->ops->creatematrix                 = DMCreateMatrix_IGA;

  dm->ops->getlocaltoglobalmapping      = DMGetLocalToGlobalMapping_IGA;
  dm->ops->globaltolocalbegin           = DMGlobalToLocalBegin_IGA;
  dm->ops->globaltolocalend             = DMGlobalToLocalEnd_IGA;
  dm->ops->localtoglobalbegin           = DMLocalToGlobalBegin_IGA;
  dm->ops->localtoglobalend             = DMLocalToGlobalEnd_IGA;
  dm->ops->localtolocalbegin            = DMLocalToLocalBegin_IGA;
  dm->ops->localtolocalend              = DMLocalToLocalEnd_IGA;
  /*
  dm->ops->refine                       = DMRefine_IGA;
  dm->ops->coarsen                      = DMCoarsen_IGA;
  dm->ops->getcoloring                  = DMCreateColoring_IGA;
  dm->ops->createinterpolation          = DMCreateInterpolation_IGA;
  dm->ops->createinjection              = DMCreateInjection_IGA;
  dm->ops->createrestriction            = DMCreateRestriction_IGA;
  */
  dm->ops->clone                        = DMClone_IGA;
  dm->ops->createcoordinatedm           = DMCreateCoordinateDM_IGA;
  dm->ops->createsubdm                  = DMCreateSubDM_IGA;
  dm->ops->createfieldis                = DMCreateFieldIS_IGA;
  dm->ops->createfielddecomposition     = DMCreateFieldDecomposition_IGA;
  /*
  dm->ops->createdomaindecomposition    = DMCreateDomainDecomposition_IGA;
  dm->ops->createddscatters             = DMCreateDomainDecompositionScatters_IGA;
  */

  PetscFunctionReturn(0);
}
EXTERN_C_END
