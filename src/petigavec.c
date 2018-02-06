#include "petiga.h"
#include <petsc/private/vecimpl.h>

#if PETSC_VERSION_LT(3,8,0)
#define PETSCVIEWERGLVIS "glvis"
#endif

static PetscErrorCode VecDuplicate_IGA(Vec g,Vec* gg)
{
  IGA            iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)g,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,gg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecView_IGA(Vec v,PetscViewer viewer)
{
  IGA            iga;
  DM             dm;
  Vec            dmvec;
  const char*    name;
  PetscBool      match;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)v,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERVTK,&match);CHKERRQ(ierr);
  if (match)  {ierr = IGADrawVec(iga,v,viewer);CHKERRQ(ierr); PetscFunctionReturn(0);}
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&match);CHKERRQ(ierr);
  if (match) {ierr = IGADrawVec(iga,v,viewer);CHKERRQ(ierr); PetscFunctionReturn(0);}
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERGLVIS,&match);CHKERRQ(ierr);
  if (match) {ierr = IGADrawVec(iga,v,viewer);CHKERRQ(ierr); PetscFunctionReturn(0);}
  ierr = IGAGetNodeDM(iga,&dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&dmvec);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)v,&name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)dmvec,name);CHKERRQ(ierr);
  ierr = VecCopy(v,dmvec);CHKERRQ(ierr);
  ierr = VecView(dmvec,viewer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&dmvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode VecLoad_IGA(Vec v,PetscViewer viewer)
{
  IGA            iga;
  DM             dm;
  Vec            dmvec;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)v,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  ierr = IGAGetNodeDM(iga,&dm);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm,&dmvec);CHKERRQ(ierr);
  ierr = VecLoad(dmvec,viewer);CHKERRQ(ierr);
  ierr = VecCopy(dmvec,v);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&dmvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(vec,2);
  IGACheckSetUpStage2(iga,1);
  ierr = VecCreate(((PetscObject)iga)->comm,vec);CHKERRQ(ierr);
  ierr = VecSetLayout(*vec,iga->map);CHKERRQ(ierr);
  ierr = VecSetType(*vec,iga->vectype);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*vec,"IGA",(PetscObject)iga);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec,VECOP_DUPLICATE,(void(*)(void))VecDuplicate_IGA);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec,VECOP_VIEW,(void(*)(void))VecView_IGA);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec,VECOP_LOAD,(void(*)(void))VecLoad_IGA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGACreateLocalVec(IGA iga,Vec *vec)
{
  PetscInt       bs,n;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(vec,2);
  IGACheckSetUpStage2(iga,1);
  /* */
  bs = iga->dof;
  n  = iga->node_gwidth[0];
  n *= iga->node_gwidth[1];
  n *= iga->node_gwidth[2];
  ierr = VecCreate(PETSC_COMM_SELF,vec);CHKERRQ(ierr);
  ierr = VecSetSizes(*vec,n*bs,n*bs);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vec,bs);CHKERRQ(ierr);
  ierr = VecSetType(*vec,iga->vectype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetLocalVec(IGA iga,Vec *lvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(lvec,2);
  IGACheckSetUpStage2(iga,1);
  if (iga->nwork > 0) {
    *lvec = iga->vwork[--iga->nwork];
    iga->vwork[iga->nwork] = NULL;
  } else {
    ierr = IGACreateLocalVec(iga,lvec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGARestoreLocalVec(IGA iga,Vec *lvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(lvec,2);
  PetscValidHeaderSpecific(*lvec,VEC_CLASSID,2);
  IGACheckSetUpStage2(iga,1);
  if (iga->nwork < (PetscInt)(sizeof(iga->vwork)/sizeof(Vec))) {
    iga->vwork[iga->nwork++] = *lvec; *lvec = NULL;
  } else {
    ierr = VecDestroy(lvec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGlobalToLocalBegin(IGA iga,Vec gvec,Vec lvec,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,3);
  IGACheckSetUpStage2(iga,1);
  ierr = VecScatterBegin(iga->g2l,gvec,lvec,addv,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGlobalToLocalEnd(IGA iga,Vec gvec,Vec lvec,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,3);
  IGACheckSetUpStage2(iga,1);
  ierr = VecScatterEnd(iga->g2l,gvec,lvec,addv,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGlobalToLocal(IGA iga,Vec gvec,Vec lvec,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAGlobalToLocalBegin(iga,gvec,lvec,addv);CHKERRQ(ierr);
  ierr = IGAGlobalToLocalEnd  (iga,gvec,lvec,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGALocalToGlobalBegin(IGA iga,Vec lvec,Vec gvec,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,3);
  IGACheckSetUpStage2(iga,1);
  if (addv == ADD_VALUES) {
    ierr = VecScatterBegin(iga->g2l,lvec,gvec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  } else if (addv == INSERT_VALUES) {
    ierr = VecScatterBegin(iga->l2g,lvec,gvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  } else SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_SUP,"Not yet implemented");
  PetscFunctionReturn(0);
}

PetscErrorCode IGALocalToGlobalEnd(IGA iga,Vec lvec,Vec gvec,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,3);
  IGACheckSetUpStage2(iga,1);
  if (addv == ADD_VALUES) {
    ierr = VecScatterEnd(iga->g2l,lvec,gvec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  } else if (addv == INSERT_VALUES) {
    ierr = VecScatterEnd(iga->l2g,lvec,gvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  } else SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_SUP,"Not yet implemented");
  PetscFunctionReturn(0);
}

PetscErrorCode IGALocalToGlobal(IGA iga,Vec lvec,Vec gvec,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGALocalToGlobalBegin(iga,lvec,gvec,addv);CHKERRQ(ierr);
  ierr = IGALocalToGlobalEnd  (iga,lvec,gvec,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGALocalToLocalBegin(IGA iga,Vec gvec,Vec lvec,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,3);
  IGACheckSetUpStage2(iga,1);
  if (!iga->l2l) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_SUP,"Not implemented");
  ierr = VecScatterBegin(iga->l2l,gvec,lvec,addv,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGALocalToLocalEnd(IGA iga,Vec gvec,Vec lvec,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(lvec,VEC_CLASSID,3);
  IGACheckSetUpStage2(iga,1);
  if (!iga->l2l) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_SUP,"Not implemented");
  ierr = VecScatterEnd(iga->l2l,gvec,lvec,addv,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGALocalToLocal(IGA iga,Vec gvec,Vec lvec,InsertMode addv)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGALocalToLocalBegin(iga,gvec,lvec,addv);CHKERRQ(ierr);
  ierr = IGALocalToLocalEnd  (iga,gvec,lvec,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetLocalVecArray(IGA iga,Vec gvec,Vec *lvec,const PetscScalar *array[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  PetscValidPointer(lvec,3);
  PetscValidPointer(array,4);
  IGACheckSetUpStage2(iga,1);
  ierr = IGAGetLocalVec(iga,lvec);CHKERRQ(ierr);
  ierr = IGAGlobalToLocal(iga,gvec,*lvec,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecGetArrayRead(*lvec,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGARestoreLocalVecArray(IGA iga,Vec gvec,Vec *lvec,const PetscScalar *array[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  PetscValidPointer(lvec,3);
  PetscValidHeaderSpecific(*lvec,VEC_CLASSID,3);
  PetscValidPointer(array,4);
  IGACheckSetUpStage2(iga,1);
  ierr = VecRestoreArrayRead(*lvec,array);CHKERRQ(ierr);
  ierr = IGARestoreLocalVec(iga,lvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetNaturalVec(IGA iga,Vec *nvec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(nvec,2);
  IGACheckSetUpStage2(iga,1);
  *nvec = iga->natural;
  PetscFunctionReturn(0);
}

PetscErrorCode IGANaturalToGlobal(IGA iga,Vec nvec,Vec gvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(nvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,3);
  IGACheckSetUpStage2(iga,1);
  ierr = VecScatterBegin(iga->n2g,nvec,gvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (iga->n2g,nvec,gvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGlobalToNatural(IGA iga,Vec gvec,Vec nvec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(gvec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(nvec,VEC_CLASSID,3);
  IGACheckSetUpStage2(iga,1);
  ierr = VecScatterBegin(iga->g2n,gvec,nvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (iga->g2n,gvec,nvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
