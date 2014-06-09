#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "IGAPointCreate"
PetscErrorCode IGAPointCreate(IGAPoint *_point)
{
  IGAPoint       point;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_point,3);
  ierr = PetscCalloc1(1,_point);CHKERRQ(ierr);
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
  point = *_point; *_point = NULL;
  if (!point) PetscFunctionReturn(0);
  if (--point->refct > 0) PetscFunctionReturn(0);
  ierr = IGAPointReset(point);CHKERRQ(ierr);
  ierr = PetscFree(point);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointReference"
PetscErrorCode IGAPointReference(IGAPoint point)
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  point->refct++;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointFreeWork"
static
PetscErrorCode IGAPointFreeWork(IGAPoint point)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  {
    size_t MAX_WORK_VEC = sizeof(point->wvec)/sizeof(PetscScalar*);
    size_t MAX_WORK_MAT = sizeof(point->wmat)/sizeof(PetscScalar*);
    size_t i;
    for (i=0; i<MAX_WORK_VEC; i++)
      {ierr = PetscFree(point->wvec[i]);CHKERRQ(ierr);}
    point->nvec = 0;
    for (i=0; i<MAX_WORK_MAT; i++)
      {ierr = PetscFree(point->wmat[i]);CHKERRQ(ierr);}
    point->nmat = 0;
  }
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
  point->count =  0;
  point->index = -1;
  point->rational = NULL;
  point->geometry = NULL;
  point->property = NULL;
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
  PetscValidPointer(element,2);
  ierr = IGAPointReset(point);CHKERRQ(ierr);
  point->parent = element;
  point->neq = element->neq;
  point->nen = element->nen;
  point->dof = element->dof;
  point->dim = element->dim;
  point->nsd = element->nsd;
  point->npd = element->npd;
  point->rational = element->rational ? element->rationalW : NULL;
  point->geometry = element->geometry ? element->geometryX : NULL;
  point->property = element->property ? element->propertyA : NULL;
  { /* */
    size_t MAX_WORK_VEC = sizeof(point->wvec)/sizeof(PetscScalar*);
    size_t MAX_WORK_MAT = sizeof(point->wmat)/sizeof(PetscScalar*);
    size_t i, nv = element->nen * element->dof, nm = nv*nv;
    for (i=0; i<MAX_WORK_VEC; i++)
      {ierr = PetscMalloc1(nv,&point->wvec[i]);CHKERRQ(ierr);}
    for (i=0; i<MAX_WORK_MAT; i++)
      {ierr = PetscMalloc1(nm,&point->wmat[i]);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetParent"
PetscErrorCode IGAPointGetParent(IGAPoint point,IGAElement *element)
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidPointer(element,2);
  *element = point->parent;
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
#define __FUNCT__ "IGAPointGetCount"
PetscErrorCode IGAPointGetCount(IGAPoint point,PetscInt *count)
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidIntPointer(count,2);
  *count = point->count;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointAtBoundary"
PetscErrorCode IGAPointAtBoundary(IGAPoint point,PetscInt *axis,PetscInt *side)
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  if (axis) PetscValidIntPointer(axis,2);
  if (side) PetscValidIntPointer(side,3);
  if (axis) *axis = point->atboundary ? point->boundary_id / 2 : -1;
  if (side) *side = point->atboundary ? point->boundary_id % 2 : -1;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetSizes"
PetscErrorCode IGAPointGetSizes(IGAPoint point,PetscInt *neq,PetscInt *nen,PetscInt *dof)
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  if (neq) PetscValidIntPointer(neq,2);
  if (nen) PetscValidIntPointer(nen,3);
  if (dof) PetscValidIntPointer(dof,4);
  if (neq) *neq = point->neq;
  if (nen) *nen = point->nen;
  if (dof) *dof = point->dof;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetDims"
PetscErrorCode IGAPointGetDims(IGAPoint point,PetscInt *dim,PetscInt *nsd,PetscInt *npd)
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  if (dim) PetscValidIntPointer(dim,2);
  if (nsd) PetscValidIntPointer(nsd,3);
  if (npd) PetscValidIntPointer(npd,4);
  if (dim) *dim = point->dim;
  if (nsd) *nsd = point->nsd;
  if (npd) *nsd = point->npd;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetQuadrature"
PetscErrorCode IGAPointGetQuadrature(IGAPoint point,
				     PetscReal *weight,
				     PetscReal *detJac)
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  if (weight) PetscValidRealPointer(weight,3);
  if (detJac) PetscValidRealPointer(detJac,4);
  if (weight) *weight = point->weight[0];
  if (detJac) *detJac = point->detJac[0];
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointGetBasisFuns"
PetscErrorCode IGAPointGetBasisFuns(IGAPoint point,PetscInt der,const PetscReal *basisfuns[])
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidPointer(basisfuns,3);
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
  PetscValidPointer(shapefuns,3);
  if (PetscUnlikely(der < 0 || der >= (PetscInt)(sizeof(point->shape)/sizeof(PetscReal*))))
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
	    "Requested derivative must be in range [0,%d], got %D",
	    (int)(sizeof(point->shape)/sizeof(PetscReal*)-1),der);
  *shapefuns = point->shape[der];
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern void IGA_GetGeomMap        (PetscInt nen,PetscInt nsd,
				   const PetscReal N[],const PetscReal C[],PetscReal X[]);
extern void IGA_GetGradGeomMap    (PetscInt nen,PetscInt nsd,PetscInt dim,
				   const PetscReal N[],const PetscReal C[],PetscReal F[]);
extern void IGA_GetInvGradGeomMap (PetscInt nen,PetscInt nsd,PetscInt dim,
				   const PetscReal N[],const PetscReal C[],PetscReal G[]);
EXTERN_C_END

#undef  __FUNCT__
#define __FUNCT__ "IGAPointFormScale"
static PetscErrorCode IGAPointFormScale(IGAPoint p,PetscReal L[])
{
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidRealPointer(L,2);
  {
    PetscInt i;
    PetscInt dim = p->dim;
    PetscInt *ID = p->parent->ID;
    IGABasis *BD = p->parent->parent->basis;
    for (i=0; i<dim; i++)
      L[i] = BD[i]->detJ[ID[i]];
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointFormGeomMap"
PetscErrorCode IGAPointFormGeomMap(IGAPoint p,PetscReal x[])
{
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidRealPointer(x,2);
  if (p->geometry) {
    const PetscReal *X = p->geometry;
    IGA_GetGeomMap(p->nen,p->nsd,p->shape[0],X,x);
  } else {
    PetscInt i,dim = p->dim;
    const PetscReal *X = p->point;
    for (i=0; i<dim; i++) x[i] = X[i];
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointFormGradGeomMap"
PetscErrorCode IGAPointFormGradGeomMap(IGAPoint p,PetscReal F[])
{
  PetscReal L[3] = {1,1,1};
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidRealPointer(F,2);
  (void)IGAPointFormScale(p,L);
  if (p->geometry) {
    PetscInt a,dim = p->dim;
    PetscInt i,nsd = p->nsd;
    if (dim == nsd) {
      (void)PetscMemcpy(F,p->gradX[0],nsd*dim*sizeof(PetscReal));
    } else {
      const PetscReal *X = p->geometry;
      IGA_GetGradGeomMap(p->nen,nsd,dim,p->basis[1],X,F);
    }
    for (i=0; i<nsd; i++)
      for (a=0; a<dim; a++)
	F[i*dim+a] *= L[a];
  } else {
    PetscInt i,dim = p->dim;
    (void)PetscMemzero(F,dim*dim*sizeof(PetscReal));
    for (i=0; i<dim; i++) F[i*(dim+1)] = L[i];
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointFormInvGradGeomMap"
PetscErrorCode IGAPointFormInvGradGeomMap(IGAPoint p,PetscReal G[])
{
  PetscReal L[3] = {1,1,1};
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidRealPointer(G,2);
  (void)IGAPointFormScale(p,L);
  if (p->geometry) {
    PetscInt a,dim = p->dim;
    PetscInt i,nsd = p->nsd;
    if (dim == nsd) {
      (void)PetscMemcpy(G,p->gradX[1],dim*nsd*sizeof(PetscReal));
    } else {
      const PetscReal *X = p->geometry;
      IGA_GetInvGradGeomMap(p->nen,nsd,dim,p->basis[1],X,G);
    }
    for (a=0; a<dim; a++)
      for (i=0; i<nsd; i++)
	G[a*nsd+i] /= L[a];
  } else {
    PetscInt i,dim = p->dim;
    (void)PetscMemzero(G,dim*dim*sizeof(PetscReal));
    for (i=0; i<dim; i++) G[i*(dim+1)] = 1/L[i];
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
void igapoint_geommap       (IGAPoint p,PetscReal x[]) {(void)IGAPointFormGeomMap(p,x);}
void igapoint_gradgeommap   (IGAPoint p,PetscReal f[]) {(void)IGAPointFormGradGeomMap(p,f);}
void igapoint_invgradgeommap(IGAPoint p,PetscReal g[]) {(void)IGAPointFormInvGradGeomMap(p,g);}
EXTERN_C_END


EXTERN_C_BEGIN
extern void IGA_GetValue(PetscInt nen,PetscInt dof,const PetscReal N[],
			 const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetGrad (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],
			 const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetHess (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],
			 const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetDel2 (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],
			 const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetDer3 (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],
			 const PetscScalar U[],PetscScalar u[]);
EXTERN_C_END

#undef  __FUNCT__
#define __FUNCT__ "IGAPointFormPoint"
PetscErrorCode IGAPointFormPoint(IGAPoint p,PetscReal x[])
{ return IGAPointFormGeomMap(p,x); }

#undef  __FUNCT__
#define __FUNCT__ "IGAPointFormValue"
PetscErrorCode IGAPointFormValue(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidScalarPointer(U,2);
  PetscValidScalarPointer(u,3);
  IGA_GetValue(p->nen,p->dof,p->shape[0],U,u);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointFormGrad"
PetscErrorCode IGAPointFormGrad(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidScalarPointer(U,2);
  PetscValidScalarPointer(u,3);
  IGA_GetGrad(p->nen,p->dof,p->dim,p->shape[1],U,u);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointFormHess"
PetscErrorCode IGAPointFormHess(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidScalarPointer(U,2);
  PetscValidScalarPointer(u,3);
  IGA_GetHess(p->nen,p->dof,p->dim,p->shape[2],U,u);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointFormDel2"
PetscErrorCode IGAPointFormDel2(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidScalarPointer(U,2);
  PetscValidScalarPointer(u,3);
  IGA_GetDel2(p->nen,p->dof,p->dim,p->shape[2],U,u);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointFormDer3"
PetscErrorCode IGAPointFormDer3(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscValidPointer(p,1);
  PetscValidScalarPointer(U,2);
  PetscValidScalarPointer(u,3);
  IGA_GetDer3(p->nen,p->dof,p->dim,p->shape[3],U,u);
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
  if (PetscUnlikely(point->nvec >= sizeof(point->wvec)/sizeof(PetscScalar*)))
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work vectors requested");
  {
    size_t m = (size_t)(point->neq * point->dof);
    *V = point->wvec[point->nvec++];
    ierr = PetscMemzero(*V,m*sizeof(PetscScalar));CHKERRQ(ierr);
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
  if (PetscUnlikely(point->nmat >= sizeof(point->wmat)/sizeof(PetscScalar*)))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work matrices requested");
  {
    size_t m = (size_t)(point->neq * point->dof);
    size_t n = (size_t)(point->nen * point->dof);
    *M = point->wmat[point->nmat++];
    ierr = PetscMemzero(*M,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
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
  (void)PetscLogFlops(2*n);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointAddVec"
PetscErrorCode IGAPointAddVec(IGAPoint point,const PetscScalar f[],PetscScalar F[])
{
  PetscInt       m;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidScalarPointer(f,2);
  PetscValidScalarPointer(F,3);
  m = point->neq * point->dof;
  ierr = IGAPointAddArray(point,m,f,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPointAddMat"
PetscErrorCode IGAPointAddMat(IGAPoint point,const PetscScalar k[],PetscScalar K[])
{
  PetscInt       m,n;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidScalarPointer(k,2);
  PetscValidScalarPointer(K,3);
  m = point->neq * point->dof;
  n = point->nen * point->dof;
  ierr = IGAPointAddArray(point,m*n,k,K);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
