#include "petiga.h"
#if PETSC_VERSION_(3,2,0)
#include <private/vecimpl.h>
#else
#include <petsc-private/vecimpl.h>
#endif

#if PETSC_VERSION_LE(3,3,0)
EXTERN_C_BEGIN
extern       PetscErrorCode VecView_MPI_DA(Vec,PetscViewer);
extern       PetscErrorCode VecLoad_Default_DA(Vec,PetscViewer);
EXTERN_C_END
#undef  __FUNCT__
#define __FUNCT__ "VecSetDM"
static PetscErrorCode VecSetDM(Vec v,DM dm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(dm,DM_CLASSID,2);
  ierr = PetscObjectCompose((PetscObject)v,"DM",(PetscObject)dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#else
PETSC_EXTERN PetscErrorCode VecView_MPI_DA(Vec,PetscViewer);
PETSC_EXTERN PetscErrorCode VecLoad_Default_DA(Vec,PetscViewer);
#endif

#undef  __FUNCT__
#define __FUNCT__ "VecDuplicate_IGA"
static PetscErrorCode VecDuplicate_IGA(Vec g,Vec* gg)
{
  IGA            iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)g,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,gg);CHKERRQ(ierr);
  ierr = PetscLayoutReference(g->map,&(*gg)->map);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "VecView_IGA"
static PetscErrorCode VecView_IGA(Vec v,PetscViewer viewer)
{
  IGA            iga;
  DM             dm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)v,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  ierr = IGAGetNodeDM(iga,&dm);CHKERRQ(ierr);
  ierr = VecSetDM(v,dm);CHKERRQ(ierr);
  ierr = VecView_MPI_DA(v,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef  __FUNCT__
#define __FUNCT__ "VecLoad_IGA"
static PetscErrorCode VecLoad_IGA(Vec v,PetscViewer viewer)
{
  IGA            iga;
  DM             dm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)v,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  ierr = IGAGetNodeDM(iga,&dm);CHKERRQ(ierr);
  ierr = VecSetDM(v,dm);CHKERRQ(ierr);
  ierr = VecLoad_Default_DA(v,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE
PetscInt Product(const PetscInt a[3]) { return a[0]*a[1]*a[2]; }

#undef  __FUNCT__
#define __FUNCT__ "IGACreateVec"
/*@
   IGACreateVec - Creates a vector with the correct parallel layout
   required for computing a vector using the discretization
   information provided in the IGA.

   Collective on IGA

   Input Parameter:
.  iga - the IGA context

   Output Parameter:
.  vec - the vector

   Level: normal

.keywords: IGA, create, vector
@*/
PetscErrorCode IGACreateVec(IGA iga, Vec *vec)
{
  PetscInt       bs,n,N;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(vec,2);
  IGACheckSetUp(iga,1);
  /* */
  bs = iga->dof;
  N  = Product(iga->node_sizes);
  n  = Product(iga->node_lwidth);
  ierr = VecCreate(((PetscObject)iga)->comm,vec);CHKERRQ(ierr);
  ierr = VecSetSizes(*vec,n*bs,N*bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vec,bs);CHKERRQ(ierr);
  ierr = VecSetType(*vec,iga->vectype);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*vec);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(*vec,iga->lgmap);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMappingBlock(*vec,iga->lgmapb);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*vec,"IGA",(PetscObject)iga);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec,VECOP_DUPLICATE,(void(*)(void))VecDuplicate_IGA);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec,VECOP_VIEW,(void(*)(void))VecView_IGA);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec,VECOP_LOAD,(void(*)(void))VecLoad_IGA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateLocalVec"
PetscErrorCode IGACreateLocalVec(IGA iga, Vec *vec)
{
  PetscInt       bs,n;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(vec,2);
  IGACheckSetUp(iga,1);
  /* */
  bs = iga->dof;
  n  = Product(iga->node_gwidth);
  ierr = VecCreate(PETSC_COMM_SELF,vec);CHKERRQ(ierr);
  ierr = VecSetSizes(*vec,n*bs,n*bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vec,bs);CHKERRQ(ierr);
  ierr = VecSetType(*vec,iga->vectype);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*vec);CHKERRQ(ierr);
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
  IGACheckSetUp(iga,1);
  if (iga->nwork > 0) {
    *lvec = iga->vwork[--iga->nwork];
    iga->vwork[iga->nwork] = NULL;
  } else {
    ierr = IGACreateLocalVec(iga,lvec);CHKERRQ(ierr);
  }
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
  IGACheckSetUp(iga,1);
  if (iga->nwork < (PetscInt)(sizeof(iga->vwork)/sizeof(Vec))) {
    iga->vwork[iga->nwork++] = *lvec; *lvec = 0;
  } else {
    ierr = VecDestroy(lvec);CHKERRQ(ierr);
  }
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
  IGACheckSetUp(iga,1);
  ierr = VecScatterBegin(iga->g2l,gvec,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (iga->g2l,gvec,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
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
  IGACheckSetUp(iga,1);
  if (addv == ADD_VALUES) {
    ierr = VecScatterBegin(iga->g2l,lvec,gvec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (iga->g2l,lvec,gvec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  } else if (addv == INSERT_VALUES) {
    ierr = VecScatterBegin(iga->l2g,lvec,gvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (iga->l2g,lvec,gvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  } else SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_SUP,"Not yet implemented");
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetLocalVecArray"
PetscErrorCode IGAGetLocalVecArray(IGA iga,Vec gvec,Vec *lvec,const PetscScalar *array[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  PetscValidPointer(lvec,3);
  PetscValidPointer(array,4);
  IGACheckSetUp(iga,1);
  ierr = IGAGetLocalVec(iga,lvec);CHKERRQ(ierr);
  ierr = IGAGlobalToLocal(iga,gvec,*lvec);CHKERRQ(ierr);
  ierr = VecGetArrayRead(*lvec,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGARestoreLocalVecArray"
PetscErrorCode IGARestoreLocalVecArray(IGA iga,Vec gvec,Vec *lvec,const PetscScalar *array[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  PetscValidPointer(lvec,3);
  PetscValidHeaderSpecific(*lvec,VEC_CLASSID,3);
  PetscValidPointer(array,4);
  IGACheckSetUp(iga,1);
  ierr = VecRestoreArrayRead(*lvec,array);CHKERRQ(ierr);
  ierr = IGARestoreLocalVec(iga,lvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetNaturalVec"
PetscErrorCode IGAGetNaturalVec(IGA iga,Vec *nvec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(nvec,2);
  IGACheckSetUp(iga,1);
  *nvec = iga->natural;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGANaturalToGlobal"
PetscErrorCode IGANaturalToGlobal(IGA iga,Vec nvec,Vec gvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(nvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,3);
  IGACheckSetUp(iga,1);
  ierr = VecScatterBegin(iga->n2g,nvec,gvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (iga->n2g,nvec,gvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGlobalToNatural"
PetscErrorCode IGAGlobalToNatural(IGA iga,Vec gvec,Vec nvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(nvec,VEC_CLASSID,3);
  IGACheckSetUp(iga,1);
  ierr = VecScatterBegin(iga->g2n,gvec,nvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (iga->g2n,gvec,nvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
