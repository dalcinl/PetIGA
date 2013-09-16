#include <petiga.h>
#include "petsc-private/dmimpl.h"

#if PETSC_VERSION_LE(3,3,0)
#undef  __FUNCT__
#define __FUNCT__ "VecSetDM"
static PetscErrorCode VecSetDM(Vec v,DM dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  if (dm) PetscValidHeaderSpecific(dm,DM_CLASSID,2);
  ierr = PetscObjectCompose((PetscObject)v,"DM",(PetscObject)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if PETSC_VERSION_(3,4,0)
#define VecSetDM(v,dm) PetscObjectCompose((PetscObject)v,"__PETSc_dm",(PetscObject)dm)
#endif

typedef struct {
  IGA iga;
} DM_IGA;

#define DMIGACast(dm) ((DM_IGA*)(dm)->data)

#undef  __FUNCT__
#define __FUNCT__ "DMIGASetIGA"
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

#undef  __FUNCT__
#define __FUNCT__ "DMIGAGetIGA"
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

#undef  __FUNCT__
#define __FUNCT__ "IGACreateWrapperDM"
PetscErrorCode IGACreateWrapperDM(IGA iga,DM *dm)
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

#undef  __FUNCT__
#define __FUNCT__ "DMIGA_GetIGA"
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

#undef  __FUNCT__
#define __FUNCT__ "DMGetLocalToGlobalMapping_IGA"
static PetscErrorCode DMGetLocalToGlobalMapping_IGA(DM dm)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  IGACheckSetUp(iga,0);
  ierr = PetscObjectReference((PetscObject)iga->lgmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&dm->ltogmap);CHKERRQ(ierr);
  dm->ltogmap = iga->lgmap;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMGetLocalToGlobalMappingBlock_IGA"
static PetscErrorCode DMGetLocalToGlobalMappingBlock_IGA(DM dm)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  IGACheckSetUp(iga,0);
  ierr = PetscObjectReference((PetscObject)iga->lgmapb);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&dm->ltogmapb);CHKERRQ(ierr);
  dm->ltogmapb = iga->lgmapb;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMGlobalToLocalBegin_IGA"
static PetscErrorCode DMGlobalToLocalBegin_IGA(DM dm,Vec g,InsertMode mode,Vec l)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAGlobalToLocalBegin(iga,g,l,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef  __FUNCT__
#define __FUNCT__ "DMGlobalToLocalBegin_IGA"
static PetscErrorCode DMGlobalToLocalEnd_IGA(DM dm,Vec g,InsertMode mode,Vec l)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAGlobalToLocalEnd(iga,g,l,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef  __FUNCT__
#define __FUNCT__ "DMLocalToGlobalBegin_IGA"
static PetscErrorCode DMLocalToGlobalBegin_IGA(DM dm,Vec l,InsertMode mode,Vec g)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGALocalToGlobalBegin(iga,l,g,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef  __FUNCT__
#define __FUNCT__ "DMLocalToGlobalBegin_IGA"
static PetscErrorCode DMLocalToGlobalEnd_IGA(DM dm,Vec l,InsertMode mode,Vec g)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGALocalToGlobalEnd(iga,l,g,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if PETSC_VERSION_LE(3,3,0)
#undef VecType
typedef const char* VecType;
#endif

#undef  __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_IGA"
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

#undef  __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_IGA"
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

#if PETSC_VERSION_LE(3,3,0)
#undef MatType
typedef const char* MatType;
#endif

#undef  __FUNCT__
#define __FUNCT__ "DMCreateMatrix_IGA"
static PetscErrorCode DMCreateMatrix_IGA(DM dm,MatType mtype,Mat *J)
{
  IGA            iga = DMIGACast(dm)->iga;
  MatType        save;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  save = iga->mattype;
  if (mtype) iga->mattype = (char*)mtype;
  ierr = IGACreateMat(iga,J);CHKERRQ(ierr);
  if (mtype) iga->mattype = (char*)save;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMDestroy_IGA"
static PetscErrorCode DMDestroy_IGA(DM dm)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFree(dm->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMSetFromOptions_IGA"
static PetscErrorCode DMSetFromOptions_IGA(DM dm)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!iga) PetscFunctionReturn(0);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMSetUp_IGA"
static PetscErrorCode DMSetUp_IGA(DM dm)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMIGA_GetIGA(dm,&iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping_IGA(dm);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMappingBlock_IGA(dm);CHKERRQ(ierr);
  ierr = IGAGetDof(iga,&dm->bs);CHKERRQ(ierr);
  ierr = DMSetVecType(dm,iga->vectype);CHKERRQ(ierr);
  ierr = DMSetMatType(dm,iga->mattype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMView_IGA"
static PetscErrorCode DMView_IGA(DM dm,PetscViewer viewer)
{
  IGA            iga = DMIGACast(dm)->iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!iga) PetscFunctionReturn(0);
  ierr = IGAView(iga,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMLoad_IGA"
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

EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "DMCreate_IGA"
PetscErrorCode DMCreate_IGA(DM dm)
{
  DM_IGA         *dd = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(dm,1);

  ierr = PetscNewLog(dm,DM_IGA,&dd);CHKERRQ(ierr);
  dm->data = dd;

#if PETSC_VERSION_LE(3,3,0)
#define getlocaltoglobalmapping      createlocaltoglobalmapping
#define getlocaltoglobalmappingblock createlocaltoglobalmappingblock
#endif
  dm->ops->getlocaltoglobalmapping      = DMGetLocalToGlobalMapping_IGA;
  dm->ops->getlocaltoglobalmappingblock = DMGetLocalToGlobalMappingBlock_IGA;
  dm->ops->globaltolocalbegin           = DMGlobalToLocalBegin_IGA;
  dm->ops->globaltolocalend             = DMGlobalToLocalEnd_IGA;
  dm->ops->localtoglobalbegin           = DMLocalToGlobalBegin_IGA;
  dm->ops->localtoglobalend             = DMLocalToGlobalEnd_IGA;
  dm->ops->createglobalvector           = DMCreateGlobalVector_IGA;
  dm->ops->createlocalvector            = DMCreateLocalVector_IGA;
  dm->ops->creatematrix                 = DMCreateMatrix_IGA;
  /*
  dm->ops->getcoloring                  = DMCreateColoring_IGA;
  dm->ops->createinterpolation          = DMCreateInterpolation_IGA;
  dm->ops->refine                       = DMRefine_IGA;
  dm->ops->coarsen                      = DMCoarsen_IGA;
  dm->ops->refinehierarchy              = DMRefineHierarchy_IGA;
  dm->ops->coarsenhierarchy             = DMCoarsenHierarchy_IGA;
  dm->ops->getinjection                 = DMCreateInjection_IGA;
  dm->ops->getaggregates                = DMCreateAggregates_IGA;
  */
  dm->ops->destroy                      = DMDestroy_IGA;
  dm->ops->view                         = DMView_IGA;
  dm->ops->setfromoptions               = DMSetFromOptions_IGA;
  dm->ops->setup                        = DMSetUp_IGA;
  dm->ops->load                         = DMLoad_IGA;
  /*
  dm->ops->createcoordinatedm           = DMCreateCoordinateDM_IGA;
  dm->ops->createsubdm                  = DMCreateSubDM_IGA;
  dm->ops->createfielddecomposition     = DMCreateFieldDecomposition_IGA;
  dm->ops->createdomaindecomposition    = DMCreateDomainDecomposition_IGA;
  dm->ops->createddscatters             = DMCreateDomainDecompositionScatters_IGA;
  */

  PetscFunctionReturn(0);
}
EXTERN_C_END
