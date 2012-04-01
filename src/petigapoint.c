#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "IGAPointCreate"
PetscErrorCode IGAPointCreate(IGAPoint *_point)
{
  IGAPoint       point;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_point,3);
  ierr = PetscNew(struct _n_IGAPoint,_point);CHKERRQ(ierr);
  point = *_point;
  point->refct =  1;
  point->index = -1;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointDestroy"
PetscErrorCode IGAPointDestroy(IGAPoint *_point)
{
  IGAPoint       point;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_point,1);
  point = *_point; *_point = 0;
  if (!point) PetscFunctionReturn(0);
  if (--point->refct > 0) PetscFunctionReturn(0);
  ierr = IGAPointReset(point);CHKERRQ(ierr);
  ierr = PetscFree(point);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointFreeWork"
static
PetscErrorCode IGAPointFreeWork(IGAPoint point)
{
  size_t i;
  size_t MAX_WORK_VEC;
  size_t MAX_WORK_MAT;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  MAX_WORK_VEC = sizeof(point->wvec)/sizeof(PetscScalar*);
  for (i=0; i<MAX_WORK_VEC; i++)
    {ierr = PetscFree(point->wvec[i]);CHKERRQ(ierr);}
  point->nvec = 0;
  MAX_WORK_MAT = sizeof(point->wmat)/sizeof(PetscScalar*);
  for (i=0; i<MAX_WORK_MAT; i++)
    {ierr = PetscFree(point->wmat[i]);CHKERRQ(ierr);}
  point->nmat = 0;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointReset"
PetscErrorCode IGAPointReset(IGAPoint point)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!point) PetscFunctionReturn(0);
  PetscValidPointer(point,1);
  point->index = -1;
  ierr = IGAPointFreeWork(point);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointSetUp"
PetscErrorCode IGAPointSetUp(IGAPoint point)
{
  IGAElement     element;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  element = point->parent;
  PetscValidPointer(element,0);
  ierr = IGAPointReset(point);CHKERRQ(ierr);

  point->nen = element->nen;
  point->dof = element->dof;
  point->dim = element->dim;

  point->count = element->nqp;
  point->index = -1;

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointBegin"
PetscErrorCode IGAPointBegin(IGAPoint point)
{
  IGAElement     element;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  element = point->parent;
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  point->index = -1;
  ierr = IGAElementBuildQuadrature(element);CHKERRQ(ierr);
  ierr = IGAElementBuildShapeFuns(element);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointNext"
PetscBool IGAPointNext(IGAPoint point)
{
  IGAElement parent = point->parent;
  PetscInt nen = point->nen;
  PetscInt dim = point->dim;
  PetscInt index;
  /* */
  point->nvec = 0;
  point->nmat = 0;
  /* */
  index = ++point->index;
  if (PetscUnlikely(index >= point->count)) {
    point->index = -1;
    return PETSC_FALSE;
  }
  /* */
  point->point    = parent->point + index * dim;
  point->weight   = parent->weight[index];
  point->detJac   = parent->detJac[index];
  point->jacobian = parent->jacobian + index * dim*dim;
  point->shape[0] = parent->shape[0] + index * nen;
  point->shape[1] = parent->shape[1] + index * nen*dim;
  point->shape[2] = parent->shape[2] + index * nen*dim*dim;
  point->shape[3] = parent->shape[3] + index * nen*dim*dim*dim;
  return PETSC_TRUE;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetParent"
PetscErrorCode IGAPointGetParent(IGAPoint point,IGAElement *parent)
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidPointer(parent,2);
  *parent = point->parent;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetIndex"
PetscErrorCode IGAPointGetIndex(IGAPoint point,PetscInt *index)
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidIntPointer(index,2);
  *index = point->index;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetSizes"
PetscErrorCode IGAPointGetSizes(IGAPoint point,PetscInt *nen,PetscInt *dof,PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  if (nen) PetscValidIntPointer(nen,2);
  if (dof) PetscValidIntPointer(dof,3);
  if (dim) PetscValidIntPointer(dim,4);
  if (nen) *nen = point->nen;
  if (dof) *dof = point->dof;
  if (dim) *dim = point->dim;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetQuadrature"
PetscErrorCode IGAPointGetQuadrature(IGAPoint pnt,PetscInt *dim,
                                     const PetscReal *point[],PetscReal *weight,PetscReal *detJac)
{
  PetscFunctionBegin;
  PetscValidPointer(pnt,1);
  if (dim)    PetscValidIntPointer(dim,2);
  if (point)  PetscValidPointer(point,3);
  if (weight) PetscValidDoublePointer(weight,4);
  if (detJac) PetscValidDoublePointer(detJac,5);
  if (dim)    *dim    = pnt->dim;
  if (point)  *point  = pnt->point;
  if (weight) *weight = pnt->weight;
  if (detJac) *detJac = pnt->detJac;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetShapeFuns"
PetscErrorCode IGAPointGetShapeFuns(IGAPoint pnt,PetscInt *nen,PetscInt *dim,
                                    const PetscReal *jacobian[],const PetscReal **shapefuns[])
{
  PetscFunctionBegin;
  PetscValidPointer(pnt,1);
  if (nen)       PetscValidIntPointer(nen,2);
  if (dim)       PetscValidIntPointer(dim,3);
  if (jacobian)  PetscValidPointer(jacobian,4);
  if (shapefuns) PetscValidPointer(shapefuns,5);
  if (nen)       *nen       = pnt->nen;
  if (dim)       *dim       = pnt->dim;
  if (jacobian)  *jacobian  = pnt->jacobian;
  if (shapefuns) *shapefuns = (const PetscReal **)pnt->shape;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern void IGA_Interpolate(PetscInt nen,PetscInt dof,PetscInt dim,PetscInt der,
                            const PetscReal N[],const PetscScalar U[],PetscScalar u[]);
EXTERN_C_END

#undef  __FUNCT__
#define __FUNCT__ "IGAPointInterpolate"
PetscErrorCode IGAPointInterpolate(IGAPoint point,PetscInt ider,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidPointer(U,2);
  PetscValidPointer(u,3);
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during point loop");
  {
    PetscInt nen = point->nen;
    PetscInt dof = point->dof;
    PetscInt dim = point->dim;
    PetscReal *N = point->shape[ider];
    IGA_Interpolate(nen,dof,dim,ider,N,U,u);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetWorkVec"
PetscErrorCode IGAPointGetWorkVec(IGAPoint point,PetscScalar *V[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidPointer(V,2);
  {
    size_t MAX_WORK_VEC = sizeof(point->wvec)/sizeof(PetscScalar*);
    PetscInt n = point->nen * point->dof;
    if (PetscUnlikely(point->index < 0))
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during point loop");
    if (PetscUnlikely(point->nvec >= (PetscInt)MAX_WORK_VEC))
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work vectors requested");
    if (PetscUnlikely(!point->wvec[point->nvec])) {
      ierr = PetscMalloc1(n,PetscScalar,&point->wvec[point->nvec]);CHKERRQ(ierr);
    }
    *V = point->wvec[point->nvec++];
    ierr = PetscMemzero(*V,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetWorkMat"
PetscErrorCode IGAPointGetWorkMat(IGAPoint point,PetscScalar *M[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidPointer(M,2);
  {
    size_t MAX_WORK_MAT = sizeof(point->wmat)/sizeof(PetscScalar*);
    PetscInt n = point->nen * point->dof;
    if (PetscUnlikely(point->nmat >= (PetscInt)MAX_WORK_MAT))
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work matrices requested");
    if (PetscUnlikely(!point->wmat[point->nmat])) {
      ierr = PetscMalloc1(n*n,PetscScalar,&point->wmat[point->nmat]);CHKERRQ(ierr);
    }
    *M = point->wmat[point->nmat++];
    ierr = PetscMemzero(*M,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointAddVec"
PetscErrorCode IGAPointAddVec(IGAPoint point,const PetscScalar f[],PetscScalar F[])
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidScalarPointer(f,2);
  PetscValidScalarPointer(F,3);
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during point loop");
  {
    PetscInt nen = point->nen;
    PetscInt dof = point->dof;
    PetscReal JW = point->detJac*point->weight;
    PetscInt i, n = nen*dof;
    for (i=0; i<n; i++) F[i] += f[i] * JW;
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointAddMat"
PetscErrorCode IGAPointAddMat(IGAPoint point,const PetscScalar k[],PetscScalar K[])
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidScalarPointer(k,2);
  PetscValidScalarPointer(K,3);
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during point loop");
  {
    PetscInt nen = point->nen;
    PetscInt dof = point->dof;
    PetscReal JW = point->detJac*point->weight;
    PetscInt i, n = (nen*dof)*(nen*dof);
    for (i=0; i<n; i++) K[i] += k[i] * JW;
  }
  PetscFunctionReturn(0);
}
