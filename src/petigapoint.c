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
#define __FUNCT__ "IGAPointInit"
PetscErrorCode IGAPointInit(IGAPoint point,IGAElement element)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidPointer(element,0);
  ierr = IGAPointReset(point);CHKERRQ(ierr);
  point->parent = element;

  point->nen = element->nen;
  point->dof = element->dof;
  point->dim = element->dim;
  point->nsd = element->nsd;

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
  IGAElement element;
  PetscInt nen = point->nen;
  PetscInt dim = point->dim;
  PetscInt nsd = point->nsd;
  PetscInt index;
  /* */
  point->nvec = 0;
  point->nmat = 0;
  /* */
  index = ++point->index;
  if (PetscUnlikely(index == 0))            goto start;
  if (PetscUnlikely(index >= point->count)) goto stop;

  point->point    += dim;
  point->weight   += 1;
  point->detJac   += 1;

  point->basis[0] += nen;
  point->basis[1] += nen*dim;
  point->basis[2] += nen*dim*dim;
  point->basis[3] += nen*dim*dim*dim;

  point->gradX    += dim*dim;
  point->shape[0] += nen;
  point->shape[1] += nen*dim;
  point->shape[2] += nen*dim*dim;
  point->shape[3] += nen*dim*dim*dim;

  return PETSC_TRUE;

 start:

  element = point->parent;

  point->point    = element->point;
  point->weight   = element->weight;
  point->detJac   = element->detJac;

  point->basis[0] = element->basis[0];
  point->basis[1] = element->basis[1];
  point->basis[2] = element->basis[2];
  point->basis[3] = element->basis[3];

  point->gradX = element->gradX;
  if (element->geometry && dim == nsd) { /* XXX */
    point->shape[0] = element->shape[0];
    point->shape[1] = element->shape[1];
    point->shape[2] = element->shape[2];
    point->shape[3] = element->shape[3];
  } else {
    point->shape[0] = element->basis[0];
    point->shape[1] = element->basis[1];
    point->shape[2] = element->basis[2];
    point->shape[3] = element->basis[3];
  }
  return PETSC_TRUE;

 stop:

  point->index = -1;
  return PETSC_FALSE;

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
PetscErrorCode IGAPointGetQuadrature(IGAPoint point,const PetscReal *qpoint[],PetscReal *weight)
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  if (qpoint) PetscValidPointer(qpoint,3);
  if (weight) PetscValidRealPointer(weight,4);
  if (qpoint) *qpoint = point->point;
  if (weight) *weight = point->weight[0];
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetJacobian"
PetscErrorCode IGAPointGetJacobian(IGAPoint point,PetscReal *detJac,const PetscReal *gradX[])
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  if (detJac) PetscValidRealPointer(detJac,2);
  if (gradX)  PetscValidRealPointer(gradX,3);
  if (detJac) *detJac   = point->detJac[0];
  if (gradX)  *gradX = point->gradX;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetBasisFuns"
PetscErrorCode IGAPointGetBasisFuns(IGAPoint point,PetscInt der,const PetscReal *basisfuns[])
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidPointer(basisfuns,2);
  if (PetscUnlikely(der < 0 || der >= (PetscInt)(sizeof(point->basis)/sizeof(PetscReal*))))
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
            "Requested derivative must be in range [0,%d], got %D",
            (int)(sizeof(point->basis)/sizeof(PetscReal*)-1),der);
  *basisfuns = point->basis[der];
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetShapeFuns"
PetscErrorCode IGAPointGetShapeFuns(IGAPoint point,PetscInt der,const PetscReal *shapefuns[])
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidPointer(shapefuns,2);
  if (PetscUnlikely(der < 0 || der >= (PetscInt)(sizeof(point->shape)/sizeof(PetscReal*))))
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
            "Requested derivative must be in range [0,%d], got %D",
            (int)(sizeof(point->shape)/sizeof(PetscReal*)-1),der);
  *shapefuns = point->shape[der];
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern void IGA_GetPoint(PetscInt nen,PetscInt dim,const PetscReal N[],
                         const PetscReal Xw[],PetscReal x[]);
extern void IGA_GetValue(PetscInt nen,PetscInt dof,const PetscReal N[],
                         const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetGrad (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],
                         const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetHess (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],
                         const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetDel2 (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],
                         const PetscScalar U[],PetscScalar u[]);
extern void IGA_Get3rdMixed (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],
			     const PetscScalar U[],PetscScalar u[]);
EXTERN_C_END

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetPoint"
PetscErrorCode IGAPointGetPoint(IGAPoint p,PetscReal x[])
{
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidRealPointer(x,2);
  if (p->parent->geometry) {
    const PetscReal *X = p->parent->geometryX;
    IGA_GetPoint(p->nen,p->dim,p->shape[0],X,x);
  } else {
    PetscInt i,dim = p->dim;
    const PetscReal *X = p->point;
    for (i=0; i<dim; i++) x[i] = X[i];
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetValue"
PetscErrorCode IGAPointGetValue(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidScalarPointer(U,2);
  PetscValidScalarPointer(u,3);
  IGA_GetValue(p->nen,p->dof,p->shape[0],U,u);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetGrad"
PetscErrorCode IGAPointGetGrad(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidScalarPointer(U,2);
  PetscValidScalarPointer(u,3);
  IGA_GetGrad(p->nen,p->dof,p->dim,p->shape[1],U,u);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetHess"
PetscErrorCode IGAPointGetHess(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidScalarPointer(U,2);
  PetscValidScalarPointer(u,3);
  IGA_GetHess(p->nen,p->dof,p->dim,p->shape[2],U,u);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetDel2"
PetscErrorCode IGAPointGetDel2(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidScalarPointer(U,2);
  PetscValidScalarPointer(u,3);
  IGA_GetDel2(p->nen,p->dof,p->dim,p->shape[2],U,u);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGet3rdMixedPartials"
PetscErrorCode IGAPointGet3rdMixedPartials(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidScalarPointer(U,2);
  PetscValidScalarPointer(u,3);
  IGA_Get3rdMixed(p->nen,p->dof,p->dim,p->shape[3],U,u);
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
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during point loop");
  {
    size_t MAX_WORK_VEC = sizeof(point->wvec)/sizeof(PetscScalar*);
    PetscInt n = point->nen * point->dof;
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
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during point loop");
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
#define __FUNCT__ "IGAPointAddArray"
PetscErrorCode IGAPointAddArray(IGAPoint point,PetscInt n,const PetscScalar a[],PetscScalar A[])
{
  PetscInt  i;
  PetscReal JW;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidScalarPointer(a,2);
  PetscValidScalarPointer(A,3);
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during point loop");
  JW = point->detJac[0] * point->weight[0];
  for (i=0; i<n; i++) A[i] += a[i] * JW;
  PetscLogFlopsNoError(2*n);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointAddVec"
PetscErrorCode IGAPointAddVec(IGAPoint point,const PetscScalar f[],PetscScalar F[])
{
  PetscInt       n;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidScalarPointer(f,2);
  PetscValidScalarPointer(F,3);
  n = point->nen*point->dof;
  ierr = IGAPointAddArray(point,n,f,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointAddMat"
PetscErrorCode IGAPointAddMat(IGAPoint point,const PetscScalar k[],PetscScalar K[])
{
  PetscInt       n;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidScalarPointer(k,2);
  PetscValidScalarPointer(K,3);
  n = point->nen*point->dof;
  ierr = IGAPointAddArray(point,n*n,k,K);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
