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

  iga->dim = -1;
  iga->dof = -1;

  for (i=0; i<3; i++) {
    ierr = IGAAxisCreate(&iga->axis[i] );CHKERRQ(ierr);
    ierr = IGARuleCreate(&iga->rule[i]);CHKERRQ(ierr);
    ierr = IGABasisCreate(&iga->basis[i]);CHKERRQ(ierr);
    ierr = IGABoundaryCreate(&iga->boundary[i][0]);CHKERRQ(ierr);
    ierr = IGABoundaryCreate(&iga->boundary[i][1]);CHKERRQ(ierr);
  }
  ierr = IGAElementCreate(&iga->iterator);CHKERRQ(ierr);

  iga->dm_geom = 0;
  iga->dm_dof  = 0;

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
  if (--((PetscObject)iga)->refct > 0) PetscFunctionReturn(0);;
  
  ierr = PetscFree(iga->userops);CHKERRQ(ierr);
  for (i=0; i<3; i++) {
    ierr = IGAAxisDestroy(&iga->axis[i]);CHKERRQ(ierr);
    ierr = IGARuleDestroy(&iga->rule[i]);CHKERRQ(ierr);
    ierr = IGABasisDestroy(&iga->basis[i]);CHKERRQ(ierr);
    ierr = IGABoundaryDestroy(&iga->boundary[i][0]);CHKERRQ(ierr);
    ierr = IGABoundaryDestroy(&iga->boundary[i][1]);CHKERRQ(ierr);
  }
  ierr = IGAElementDestroy(&iga->iterator);CHKERRQ(ierr);

  ierr = DMDestroy(&iga->dm_geom);CHKERRQ(ierr);
  ierr = DMDestroy(&iga->dm_dof);CHKERRQ(ierr);

  ierr = PetscHeaderDestroy(&iga);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAView"
PetscErrorCode IGAView(IGA iga,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)iga)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
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
PetscErrorCode IGASetDof(IGA iga,PetscInt dof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,dof,2);
  if (iga->dof > 0 && iga->dof != dof)
    SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
             "Cannot change IGA dof from %D after it was set to %D",iga->dof,dof);
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
#define __FUNCT__ "IGAGetBoundary"
PetscErrorCode IGAGetBoundary(IGA iga,PetscInt i,PetscInt side,IGABoundary *boundary)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(boundary,4);
  if (iga->dim <= 0) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetDim() first");
  if (iga->dof <= 0) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetDof() first");
  if (i < 0)         SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D must be nonnegative",i);
  if (i >= iga->dim) SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Index %D, but dimension %D",i,iga->dim);
  if (side < 0) side = 0; /* XXX error ?*/
  if (side > 1) side = 1; /* XXX error ?*/
  if (iga->boundary[i][side]->dof < 1) {
    ierr = IGABoundaryInit(iga->boundary[i][side],iga->dof);CHKERRQ(ierr);
  }
  *boundary = iga->boundary[i][side];
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetFromOptions"
PetscErrorCode IGASetFromOptions(IGA iga)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)iga);CHKERRQ(ierr);
  {

  }
  ierr = PetscObjectProcessOptionsHandlers((PetscObject)iga);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASetUp"
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

  if (iga->dof < 1)
    iga->dof = 1;

  for (i=0; i<3; i++) {
    if (!iga->axis[i])  {ierr = IGAAxisCreate(&iga-> axis[i]);CHKERRQ(ierr);}
    if (!iga->rule[i])  {ierr = IGARuleCreate(&iga->rule[i]);CHKERRQ(ierr);}
    if (!iga->basis[i]) {ierr = IGABasisCreate(&iga->basis[i]);CHKERRQ(ierr);}
  }
  if (!iga->iterator) {
    ierr = IGAElementCreate(&iga->iterator);CHKERRQ(ierr);
  }
  
  for (i=0; i<iga->dim; i++) {
    ierr = IGAAxisCheck(iga->axis[i]);CHKERRQ(ierr);
  }
  for (i=0; i<3; i++) {
    if (!iga->rule[i]->nqp) {
      PetscInt p = iga->axis[i]->p, q = p*p/2+1;
      ierr = IGARuleInit(iga->rule[i],q);CHKERRQ(ierr);
    }
    if (!iga->basis[i]->nel) {
      PetscInt d = 3; /* XXX */
      ierr = IGABasisInit(iga->basis[i],iga->axis[i],iga->rule[i],d);CHKERRQ(ierr);
    }
  }

  {
    PetscInt swidth, sizes[3];
    DMDABoundaryType btype[3];
    for (i=0; i<3; i++) {
      swidth = 0; sizes[i] = 1;
      btype[i] = DMDA_BOUNDARY_NONE;
    }
    for (i=0; i<iga->dim; i++) {
      IGAAxis  axis = iga->axis[i];
      PetscInt p = axis->p;
      PetscInt m = axis->m;
      PetscInt n = m-p-1;
      swidth = PetscMax(swidth,p);
      sizes[i] = axis->periodic ? n+1-p : n+1;
      btype[i] = axis->periodic ? DMDA_BOUNDARY_PERIODIC : DMDA_BOUNDARY_NONE;
    }
    ierr = DMDACreate(((PetscObject)iga)->comm,&iga->dm_dof);CHKERRQ(ierr);
    ierr = DMDASetDim(iga->dm_dof,iga->dim); CHKERRQ(ierr);
    ierr = DMDASetDof(iga->dm_dof,iga->dof); CHKERRQ(ierr);
    ierr = DMDASetSizes(iga->dm_dof,sizes[0],sizes[1],sizes[2]); CHKERRQ(ierr);
    ierr = DMDASetBoundaryType(iga->dm_dof,btype[0],btype[1],btype[2]); CHKERRQ(ierr);
    ierr = DMDASetStencilType(iga->dm_dof,DMDA_STENCIL_BOX); CHKERRQ(ierr);
    ierr = DMDASetStencilWidth(iga->dm_dof,swidth); CHKERRQ(ierr);
    /*ierr = DMSetOptionsPrefix(iga->dm_dof, "dof_"); CHKERRQ(ierr);*/
    /*ierr = DMSetFromOptions(iga->dm_dof); CHKERRQ(ierr);*/
    ierr = DMSetUp(iga->dm_dof);CHKERRQ(ierr);
  }

  iga->setup = PETSC_TRUE;

  iga->iterator->parent = iga;
  ierr = IGAElementSetUp(iga->iterator);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetDofDM"
PetscErrorCode IGAGetDofDM(IGA iga, DM *dm_dof)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  *dm_dof = iga->dm_dof;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetGeomDM"
PetscErrorCode IGAGetGeomDM(IGA iga, DM *dm_geom)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  *dm_geom = iga->dm_geom;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateVec"
PetscErrorCode IGACreateVec(IGA iga, Vec *vec)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(vec,2);
  ierr = DMCreateGlobalVector(iga->dm_dof,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGACreateMat"
PetscErrorCode IGACreateMat(IGA iga, Mat *mat)
{
  const MatType mat_type = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(mat,2);
  mat_type = (iga->dof > 1) ? MATBAIJ : MATAIJ;
  ierr = DMGetMatrix(iga->dm_dof,mat_type,mat);CHKERRQ(ierr);
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
  ierr = DMLocalToGlobalBegin(iga->dm_dof,lvec,addv,gvec);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (iga->dm_dof,lvec,addv,gvec);CHKERRQ(ierr);
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
