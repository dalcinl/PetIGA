#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "IGACreateVec"
PetscErrorCode IGACreateVec(IGA iga, Vec *vec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(vec,2);
  IGACheckSetUp(iga,1);
  ierr = DMCreateGlobalVector(iga->dm_dof,vec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)*vec,"IGA",(PetscObject)iga);CHKERRQ(ierr);
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
  IGACheckSetUp(iga,1);
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
  IGACheckSetUp(iga,1);
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
  IGACheckSetUp(iga,1);
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
  IGACheckSetUp(iga,1);
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
  IGACheckSetUp(iga,1);
  ierr = DMLocalToGlobalBegin(iga->dm_dof,lvec,addv,gvec);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (iga->dm_dof,lvec,addv,gvec);CHKERRQ(ierr);
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
  ierr = VecRestoreArrayRead(*lvec,array);
  ierr = IGARestoreLocalVec(iga,lvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
